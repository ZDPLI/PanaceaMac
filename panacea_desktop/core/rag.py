from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .utils import tokenize, chunk_text
from .db import Database


@dataclass
class RetrievalChunk:
    chunk_id: int
    doc_id: int
    title: str
    score: float
    text: str


def add_document_to_rag(
    db: Database,
    file_path: str,
    title: str | None = None,
    *,
    scope: str = "global",
    dialog_id: int | None = None,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> int:
    """Ingest a document into RAG.

    Copies the file into app data folder for stability, extracts text, chunks it,
    and stores chunks in SQLite.
    """
    from .doc_extract import extract_text_from_file
    from .file_store import import_file

    stored = import_file(file_path, category="rag_docs")
    text = extract_text_from_file(stored.storage_path)
    if not text.strip():
        raise ValueError("Document appears to be empty after extraction")

    if title is None:
        title = stored.original_name

    doc_id = db.rag_add_doc(
        title=title,
        path=stored.storage_path,
        enabled=True,
        scope=scope,
        dialog_id=dialog_id,
        sha256=stored.sha256,
        mime=stored.mime,
        size=stored.size,
        storage_path=stored.storage_path,
    )
    chunks = chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
    db.rag_add_chunks(doc_id, chunks)
    return doc_id


def _retrieve_lexical(db: Database, query: str, *, top_k: int = 6, dialog_id: int | None = None) -> list[RetrievalChunk]:
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    # Get enabled chunks
    chunks = db.rag_get_enabled_chunks(dialog_id=dialog_id)
    if not chunks:
        return []

    # Compute document frequency for query terms (simple IDF)
    df: dict[str, int] = {t: 0 for t in set(q_tokens)}
    chunk_tokens_cache: dict[int, list[str]] = {}
    for ch in chunks:
        toks = tokenize(ch["text"])
        chunk_tokens_cache[ch["chunk_id"]] = toks
        st = set(toks)
        for t in df:
            if t in st:
                df[t] += 1
    N = len(chunks)
    idf = {t: math.log((N + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}

    # Score chunks
    scores: list[RetrievalChunk] = []
    for ch in chunks:
        toks = chunk_tokens_cache[ch["chunk_id"]]
        if not toks:
            continue
        tf: dict[str, int] = {}
        for t in toks:
            if t in idf:
                tf[t] = tf.get(t, 0) + 1
        score = 0.0
        for t in q_tokens:
            if t in tf:
                score += idf.get(t, 0.0) * (1.0 + math.log(1 + tf[t]))
        if score <= 0:
            continue
        scores.append(
            RetrievalChunk(
                chunk_id=ch["chunk_id"],
                doc_id=ch["doc_id"],
                title=ch["title"],
                score=score,
                text=ch["text"],
            )
        )

    scores.sort(key=lambda x: x.score, reverse=True)
    return scores[:top_k]


def _retrieve_bm25(db: Database, query: str, *, top_k: int = 6, dialog_id: int | None = None) -> list[RetrievalChunk]:
    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        # Fallback to lexical if bm25 lib missing
        return _retrieve_lexical(db, query, top_k=top_k, dialog_id=dialog_id)

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    chunks = db.rag_get_enabled_chunks(dialog_id=dialog_id)
    if not chunks:
        return []

    corpus_tokens: list[list[str]] = [tokenize(ch["text"]) for ch in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(q_tokens)

    out: list[RetrievalChunk] = []
    for ch, sc in zip(chunks, scores, strict=False):
        if sc <= 0:
            continue
        out.append(
            RetrievalChunk(
                chunk_id=int(ch["chunk_id"]),
                doc_id=int(ch["doc_id"]),
                title=str(ch["title"]),
                score=float(sc),
                text=str(ch["text"]),
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:top_k]


def _app_data_dir() -> Path:
    # Keep FAISS artifacts in the same per-user app data dir as the DB/files.
    # Import locally to avoid import cycles during module import in some frozen builds.
    from .db import _app_data_dir as _db_app_data_dir

    d = _db_app_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _faiss_paths() -> tuple[Path, Path]:
    d = _app_data_dir()
    return d / "rag_faiss.index", d / "rag_faiss_map.json"


def _ensure_faiss_index(db: Database, *, embedding_model: str, dialog_id: int | None = None) -> tuple["faiss.Index", list[int]]:
    """Build or load FAISS index for enabled chunks.

    Persists index and mapping in app data dir. Uses `rag_index_dirty` setting to decide rebuild.
    """
    try:
        import faiss  # type: ignore
        import numpy as np
    except Exception as e:
        raise RuntimeError("FAISS mode requires 'faiss-cpu' and 'numpy' installed") from e

    index_path, map_path = _faiss_paths()
    dirty = (db.get_setting("rag_index_dirty", "1") or "1") == "1"

    if not dirty and index_path.exists() and map_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            mapping = json.loads(map_path.read_text(encoding="utf-8"))
            chunk_ids = [int(x) for x in mapping.get("chunk_ids", [])]
            meta_model = mapping.get("embedding_model")
            if meta_model != embedding_model:
                dirty = True
            else:
                return index, chunk_ids
        except Exception:
            dirty = True

    # Rebuild
    chunks = db.rag_get_enabled_chunks(dialog_id=dialog_id)
    if not chunks:
        # Empty index
        index = faiss.IndexFlatIP(384)
        map_path.write_text(json.dumps({"chunk_ids": [], "embedding_model": embedding_model}, ensure_ascii=False), encoding="utf-8")
        faiss.write_index(index, str(index_path))
        db.set_setting("rag_index_dirty", "0")
        return index, []

    # Load embedding model lazily
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError("FAISS mode requires 'sentence-transformers' installed") from e

    model = SentenceTransformer(embedding_model)

    # Compute embeddings (use DB cache when possible)
    vectors: list["np.ndarray"] = []
    chunk_ids: list[int] = []
    for ch in chunks:
        cid = int(ch["chunk_id"])
        blob = db.rag_get_embedding(cid, embedding_model)
        if blob is None:
            emb = model.encode(ch["text"], normalize_embeddings=True)
            vec = np.asarray(emb, dtype="float32")
            db.rag_upsert_embedding(cid, embedding_model, vec.tobytes())
        else:
            vec = np.frombuffer(blob, dtype="float32")
            # ensure normalized
            nrm = float(np.linalg.norm(vec) + 1e-12)
            vec = (vec / nrm).astype("float32")
        vectors.append(vec)
        chunk_ids.append(cid)

    mat = np.vstack(vectors).astype("float32")
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    faiss.write_index(index, str(index_path))
    map_path.write_text(json.dumps({"chunk_ids": chunk_ids, "embedding_model": embedding_model}, ensure_ascii=False), encoding="utf-8")
    db.set_setting("rag_index_dirty", "0")
    return index, chunk_ids


def _retrieve_faiss(db: Database, query: str, *, top_k: int = 6, embedding_model: str, dialog_id: int | None = None) -> list[RetrievalChunk]:
    try:
        import numpy as np
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        # If deps missing, fallback
        return _retrieve_bm25(db, query, top_k=top_k, dialog_id=dialog_id)

    q = (query or "").strip()
    if not q:
        return []

    index, chunk_ids = _ensure_faiss_index(db, embedding_model=embedding_model, dialog_id=dialog_id)
    if not chunk_ids:
        return []

    model = SentenceTransformer(embedding_model)
    qvec = model.encode(q, normalize_embeddings=True)
    qvec = np.asarray(qvec, dtype="float32")[None, :]

    scores, idxs = index.search(qvec, min(top_k, len(chunk_ids)))
    idxs_list = idxs[0].tolist()
    scores_list = scores[0].tolist()

    # Map index positions to chunk ids and fetch texts
    by_id = {int(ch['chunk_id']): ch for ch in db.rag_get_enabled_chunks(dialog_id=dialog_id)}
    out: list[RetrievalChunk] = []
    for pos, sc in zip(idxs_list, scores_list, strict=False):
        if pos < 0 or pos >= len(chunk_ids):
            continue
        cid = chunk_ids[pos]
        ch = by_id.get(cid)
        if not ch:
            continue
        out.append(
            RetrievalChunk(
                chunk_id=cid,
                doc_id=int(ch["doc_id"]),
                title=str(ch["title"]),
                score=float(sc),
                text=str(ch["text"]),
            )
        )
    out.sort(key=lambda x: x.score, reverse=True)
    return out[:top_k]


def retrieve(db: Database, query: str, *, top_k: int = 6, dialog_id: int | None = None) -> list[RetrievalChunk]:
    mode = (db.get_setting("rag_retrieval_mode", "bm25") or "bm25").lower().strip()
    if mode == "faiss":
        emb_model = db.get_setting("rag_embedding_model", "sentence-transformers/all-MiniLM-L6-v2") or "sentence-transformers/all-MiniLM-L6-v2"
        return _retrieve_faiss(db, query, top_k=top_k, embedding_model=emb_model, dialog_id=dialog_id)
    if mode == "lexical":
        return _retrieve_lexical(db, query, top_k=top_k, dialog_id=dialog_id)
    # default bm25
    return _retrieve_bm25(db, query, top_k=top_k, dialog_id=dialog_id)


def build_rag_context(db: Database, query: str, *, top_k: int = 6, max_chars: int = 6000, dialog_id: int | None = None) -> str:
    """Build a RAG context block.

    Primary path: lexical retrieval over enabled chunks.
    Fallback: if the user asks *about the attached/uploaded document* but the query
    is too generic for lexical retrieval, include leading chunks from the most
    recently added enabled document.
    """
    query = (query or "").strip()
    res = retrieve(db, query, top_k=top_k, dialog_id=dialog_id)

    # Heuristic fallback for generic queries like "Что в приложенном документе?"
    if not res:
        q = query.lower()
        generic_markers = (
            "приложен", "в документ", "в файле", "в pdf", "в docx", "в приложении",
            "в загруж", "в прикреп", "что изложено", "что написано", "о чем документ",
        )
        if any(k in q for k in generic_markers):
            doc = db.rag_latest_enabled_doc(dialog_id=dialog_id)
            if doc:
                # Take the first N chunks (acts like 'read the document head')
                chunks = db.rag_get_doc_chunks(int(doc["id"]), limit=top_k)
                res = [RetrievalChunk(
                    chunk_id=int(ch["chunk_id"]),
                    doc_id=int(ch["doc_id"]),
                    title=str(ch.get("title") or doc.get("title") or "Document"),
                    text=str(ch.get("text") or ""),
                    score=0.0,
                ) for ch in chunks if (ch.get("text") or "").strip()]

    if not res:
        return ""

    parts: list[str] = []
    total = 0
    for i, ch in enumerate(res, 1):
        block = f"[Источник {i}: {ch.title}]\n{ch.text.strip()}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n\n".join(parts).strip()
