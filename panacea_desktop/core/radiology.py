from __future__ import annotations

"""Radiology / imaging support (CT/MRI).

Design goals (desktop MVP):
- Import study as DICOM folder/ZIP or NIfTI.
- If multiple DICOM series exist, allow selecting a specific SeriesInstanceUID.
- Generate a small set of preview PNG images.
- Persist previews in the app file store and register in DB.
- Inject a [RADIOLOGY CONTEXT] block + preview images into the LLM request.

Important:
This is NOT a diagnostic engine. It prepares previews + structured context.
"""

import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from .db import Database
from .file_store import import_file


ProgressCB = Callable[[int, str, str], None]


@dataclass
class IngestResult:
    study_id: int
    n_previews: int
    modality: str
    series_uid: str


def _safe_str(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return ""


def _zip_dir(src_dir: Path, dst_zip: Path) -> None:
    """Zip a directory recursively."""
    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(src_dir)))


def _to_uint8_windowed(arr: np.ndarray, *, wl: float, ww: float) -> np.ndarray:
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    a = np.clip(arr.astype(np.float32), lo, hi)
    a = (a - lo) / max(hi - lo, 1e-6)
    return (a * 255.0).astype(np.uint8)


def _to_uint8_percentile(arr: np.ndarray, *, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    a = arr.astype(np.float32)
    lo = float(np.percentile(a, p_lo))
    hi = float(np.percentile(a, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(a.min()), float(a.max())
        if hi <= lo:
            return np.zeros_like(a, dtype=np.uint8)
    a = np.clip(a, lo, hi)
    a = (a - lo) / max(hi - lo, 1e-6)
    return (a * 255.0).astype(np.uint8)


def _save_png_uint8(img2d: np.ndarray, out_path: Path) -> None:
    from PIL import Image

    if img2d.ndim != 2:
        raise ValueError("Expected 2D image")
    Image.fromarray(img2d, mode="L").save(str(out_path), format="PNG")


# -------------------- DICOM scanning helpers --------------------

def _dicom_read_header_from_bytes(data: bytes) -> Any:
    try:
        import pydicom
    except Exception as e:
        raise RuntimeError("pydicom is required for DICOM import. Please install requirements.") from e

    return pydicom.dcmread(
        pydicom.filebase.DicomBytesIO(data),
        stop_before_pixels=True,
        force=True,
        specific_tags=[
            "SeriesInstanceUID",
            "StudyInstanceUID",
            "Modality",
            "BodyPartExamined",
            "SeriesDescription",
            "StudyDescription",
            "InstanceNumber",
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "PixelSpacing",
            "SliceThickness",
            "SpacingBetweenSlices",
            "Rows",
            "Columns",
        ],
    )


def _dicom_read_header_from_path(path: Path) -> Any:
    try:
        import pydicom
    except Exception as e:
        raise RuntimeError("pydicom is required for DICOM import. Please install requirements.") from e

    return pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
        force=True,
        specific_tags=[
            "SeriesInstanceUID",
            "StudyInstanceUID",
            "Modality",
            "BodyPartExamined",
            "SeriesDescription",
            "StudyDescription",
            "InstanceNumber",
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "PixelSpacing",
            "SliceThickness",
            "SpacingBetweenSlices",
            "Rows",
            "Columns",
        ],
    )


def _series_summary_from_items(sid: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "series_uid": sid,
            "slices": 0,
            "modality": "",
            "series_desc": "",
            "study_desc": "",
            "body_part": "",
            "rows": 0,
            "cols": 0,
        }
    rows = max(int(i.get("rows", 0) or 0) for i in items)
    cols = max(int(i.get("cols", 0) or 0) for i in items)
    modality = _safe_str(items[0].get("modality", "") or "").upper()
    return {
        "series_uid": sid,
        "slices": int(len(items)),
        "modality": modality,
        "series_desc": _safe_str(items[0].get("series_desc", "") or ""),
        "study_desc": _safe_str(items[0].get("study_desc", "") or ""),
        "body_part": _safe_str(items[0].get("body", "") or ""),
        "rows": rows,
        "cols": cols,
    }


def _dicom_series_index_from_zip(zf: zipfile.ZipFile, *, limit_files: int = 50000) -> dict[str, list[dict[str, Any]]]:
    """Scan DICOM headers in a ZIP and group by SeriesInstanceUID."""
    series: dict[str, list[dict[str, Any]]] = {}

    names = [n for n in zf.namelist() if not n.endswith("/")]
    if len(names) > limit_files:
        names = names[:limit_files]

    for name in names:
        low = name.lower()
        if low.endswith((".txt", ".json", ".xml", ".csv")):
            continue
        try:
            data = zf.read(name)
        except Exception:
            continue
        if len(data) < 256:
            continue
        try:
            ds = _dicom_read_header_from_bytes(data)
        except Exception:
            continue

        sid = _safe_str(getattr(ds, "SeriesInstanceUID", "") or "")
        if not sid:
            continue

        entry = {
            "name": name,
            "instance": int(getattr(ds, "InstanceNumber", 0) or 0),
            "ipp": getattr(ds, "ImagePositionPatient", None),
            "modality": _safe_str(getattr(ds, "Modality", "") or ""),
            "body": _safe_str(getattr(ds, "BodyPartExamined", "") or ""),
            "series_desc": _safe_str(getattr(ds, "SeriesDescription", "") or ""),
            "study_desc": _safe_str(getattr(ds, "StudyDescription", "") or ""),
            "rows": int(getattr(ds, "Rows", 0) or 0),
            "cols": int(getattr(ds, "Columns", 0) or 0),
            "pixel_spacing": getattr(ds, "PixelSpacing", None),
            "slice_thickness": _safe_str(getattr(ds, "SliceThickness", "") or ""),
            "spacing_between": _safe_str(getattr(ds, "SpacingBetweenSlices", "") or ""),
        }
        series.setdefault(sid, []).append(entry)

    return series


def _dicom_series_index_from_dir(src_dir: Path, *, limit_files: int = 50000) -> dict[str, list[dict[str, Any]]]:
    """Scan a DICOM directory recursively and group by SeriesInstanceUID."""
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(str(src_dir))

    series: dict[str, list[dict[str, Any]]] = {}
    files: list[Path] = []
    for p in src_dir.rglob("*"):
        if p.is_file():
            files.append(p)
            if len(files) >= limit_files:
                break

    for p in files:
        low = p.name.lower()
        if low.endswith((".txt", ".json", ".xml", ".csv")):
            continue
        # Very small files are unlikely to be DICOM.
        try:
            if p.stat().st_size < 256:
                continue
        except Exception:
            continue
        try:
            ds = _dicom_read_header_from_path(p)
        except Exception:
            continue

        sid = _safe_str(getattr(ds, "SeriesInstanceUID", "") or "")
        if not sid:
            continue
        entry = {
            "name": str(p),
            "instance": int(getattr(ds, "InstanceNumber", 0) or 0),
            "ipp": getattr(ds, "ImagePositionPatient", None),
            "modality": _safe_str(getattr(ds, "Modality", "") or ""),
            "body": _safe_str(getattr(ds, "BodyPartExamined", "") or ""),
            "series_desc": _safe_str(getattr(ds, "SeriesDescription", "") or ""),
            "study_desc": _safe_str(getattr(ds, "StudyDescription", "") or ""),
            "rows": int(getattr(ds, "Rows", 0) or 0),
            "cols": int(getattr(ds, "Columns", 0) or 0),
            "pixel_spacing": getattr(ds, "PixelSpacing", None),
            "slice_thickness": _safe_str(getattr(ds, "SliceThickness", "") or ""),
            "spacing_between": _safe_str(getattr(ds, "SpacingBetweenSlices", "") or ""),
        }
        series.setdefault(sid, []).append(entry)

    return series


def list_dicom_series(source_path: str, *, limit_files: int = 50000) -> list[dict[str, Any]]:
    """List DICOM series for a directory or a DICOM ZIP.

    Returns a list of dicts with keys:
      series_uid, slices, modality, series_desc, study_desc, body_part, rows, cols

    Raises if source_path isn't a directory or zip.
    """
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(str(src))

    if src.is_dir():
        idx = _dicom_series_index_from_dir(src, limit_files=limit_files)
    else:
        if src.suffix.lower() != ".zip":
            raise RuntimeError("Series listing is supported only for DICOM folders or .zip")
        with zipfile.ZipFile(str(src), "r") as zf:
            idx = _dicom_series_index_from_zip(zf, limit_files=limit_files)

    out = [_series_summary_from_items(sid, items) for sid, items in idx.items()]
    # Sort: most slices first
    out.sort(key=lambda d: (int(d.get("slices", 0) or 0), int(d.get("rows", 0) or 0) * int(d.get("cols", 0) or 0)), reverse=True)
    return out


# -------------------- Compatibility wrappers --------------------
# Some UI code (qt_app.py) imports `scan_dicom_series`. During the refactor we renamed it
# to `list_dicom_series`. Keep a thin wrapper to avoid breaking already built binaries.
def scan_dicom_series(
    source_path: str,
    *,
    limit_files: int = 50000,
    progress_cb: Optional[ProgressCB] = None,
) -> list[dict[str, Any]]:
    """Scan DICOM source and return discovered series.

    Args:
        source_path: Directory with DICOM files (recursively) OR a DICOM .zip.
        limit_files: Safety limit on number of files to scan.
        progress_cb: Optional callback (pct, message, preview_path).

    Returns:
        Same format as `list_dicom_series`.
    """

    def _p(pct: int, msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(int(max(0, min(100, pct))), str(msg or ""), "")
        except Exception:
            pass

    _p(5, "Сканирую серии DICOM…")
    series = list_dicom_series(source_path, limit_files=limit_files)
    _p(100, f"Найдено серий: {len(series)}")
    return series


def _choose_best_series(series_index: dict[str, list[dict[str, Any]]]) -> tuple[str, list[dict[str, Any]]]:
    if not series_index:
        return "", []
    best_sid = ""
    best_items: list[dict[str, Any]] = []
    best_score = -1
    for sid, items in series_index.items():
        if not items:
            continue
        rows = max(int(i.get("rows", 0) or 0) for i in items)
        cols = max(int(i.get("cols", 0) or 0) for i in items)
        n = len(items)
        score = n * 10 + int(rows * cols / 10000)
        if score > best_score:
            best_score = score
            best_sid = sid
            best_items = items
    return best_sid, best_items


def _sort_slices(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def zpos(it: dict[str, Any]) -> float:
        ipp = it.get("ipp")
        try:
            if ipp is not None and len(ipp) >= 3:
                return float(ipp[2])
        except Exception:
            pass
        return float(it.get("instance", 0) or 0)

    return sorted(items, key=zpos)


def _load_dicom_pixels_from_zip(zf: zipfile.ZipFile, name: str) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import pydicom
    except Exception as e:
        raise RuntimeError("pydicom is required for DICOM import. Please install requirements.") from e

    data = zf.read(name)
    ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(data), force=True)

    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    modality = _safe_str(getattr(ds, "Modality", "") or "")
    if modality.upper() == "CT":
        arr = arr * slope + intercept

    meta = {
        "Modality": modality,
        "SeriesDescription": _safe_str(getattr(ds, "SeriesDescription", "") or ""),
        "StudyDescription": _safe_str(getattr(ds, "StudyDescription", "") or ""),
        "BodyPartExamined": _safe_str(getattr(ds, "BodyPartExamined", "") or ""),
        "Rows": int(getattr(ds, "Rows", 0) or 0),
        "Columns": int(getattr(ds, "Columns", 0) or 0),
        "PixelSpacing": list(getattr(ds, "PixelSpacing", [])) if getattr(ds, "PixelSpacing", None) is not None else [],
        "SliceThickness": _safe_str(getattr(ds, "SliceThickness", "") or ""),
        "SpacingBetweenSlices": _safe_str(getattr(ds, "SpacingBetweenSlices", "") or ""),
        "RescaleSlope": slope,
        "RescaleIntercept": intercept,
    }
    return arr, meta


# -------------------- Ingest / previews --------------------

def ingest_imaging_study(
    db: Database,
    *,
    dialog_id: int,
    source_path: str,
    modality: str = "",
    body_region: str = "",
    contrast: str = "",
    clinical_question: str = "",
    series_uid: str | None = None,
    max_previews: int = 12,
    progress_cb: Optional[ProgressCB] = None,
) -> IngestResult:
    """Import a study and generate preview images.

    - If source_path is a directory: it will be zipped and stored.
    - If it's a .zip: treated as a DICOM ZIP.
    - If it's .nii/.nii.gz: treated as NIfTI.

    If series_uid is set (DICOM only), we will prefer that SeriesInstanceUID.
    """

    def _progress(pct: int, msg: str, preview_path: str = "") -> None:
        try:
            if progress_cb is not None:
                progress_cb(int(max(0, min(100, pct))), str(msg or ""), str(preview_path or ""))
        except Exception:
            pass

    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(str(src))

    # Normalize source into a single stored file in app data.
    source_type = "unknown"
    stored_path = ""

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        if src.is_dir():
            _progress(2, "Упаковываю DICOM-папку в ZIP…")
            tmp_zip = td_p / (src.name + ".zip")
            _zip_dir(src, tmp_zip)
            _progress(8, "Сохраняю ZIP в хранилище приложения…")
            stored = import_file(str(tmp_zip), category="imaging")
            stored_path = stored.storage_path
            source_type = "dicom_dir"
        else:
            low = src.name.lower()
            if src.suffix.lower() == ".zip":
                _progress(5, "Сохраняю DICOM ZIP в хранилище приложения…")
                stored = import_file(str(src), category="imaging")
                stored_path = stored.storage_path
                source_type = "dicom_zip"
            elif low.endswith(".nii") or low.endswith(".nii.gz"):
                _progress(5, "Сохраняю NIfTI в хранилище приложения…")
                stored = import_file(str(src), category="imaging")
                stored_path = stored.storage_path
                source_type = "nifti"
            else:
                _progress(5, "Сохраняю файл в хранилище приложения…")
                stored = import_file(str(src), category="imaging")
                stored_path = stored.storage_path
                source_type = "unknown"

        # Create study row early; we will fill meta and previews below.
        study_meta: dict[str, Any] = {}
        if series_uid:
            study_meta["preferred_series_uid"] = str(series_uid)

        study_id = db.imaging_create_study(
            dialog_id=dialog_id,
            modality=modality,
            body_region=body_region,
            contrast=contrast,
            clinical_question=clinical_question,
            source_type=source_type,
            source_storage_path=stored_path,
            meta=study_meta,
        )

        n_prev = 0
        inferred_modality = ""
        selected_series_uid = ""

        if source_type in {"dicom_zip", "dicom_dir"} and stored_path.lower().endswith(".zip"):
            _progress(12, "Сканирую DICOM и выбираю серию…")
            n_prev, inferred_modality, extracted_meta = _generate_previews_from_dicom_zip(
                db,
                study_id=study_id,
                zip_path=stored_path,
                series_uid=series_uid,
                max_previews=max_previews,
                progress_cb=_progress,
            )
            selected_series_uid = str(extracted_meta.get("series_uid") or "")
            if inferred_modality and not modality:
                modality = inferred_modality
            study_meta.update(extracted_meta)
        elif source_type == "nifti":
            _progress(12, "Читаю NIfTI и готовлю preview…")
            n_prev, extracted_meta = _generate_previews_from_nifti(
                db,
                study_id=study_id,
                nifti_path=stored_path,
                max_previews=max_previews,
                progress_cb=_progress,
            )
            selected_series_uid = ""
            study_meta.update(extracted_meta)

        # Persist meta update (best-effort)
        try:
            with db._lock, db._conn() as conn:  # noqa: SLF001
                cur = conn.cursor()
                cur.execute(
                    "UPDATE imaging_studies SET modality=COALESCE(NULLIF(?,''), modality), body_region=COALESCE(NULLIF(?,''), body_region), "
                    "contrast=COALESCE(NULLIF(?,''), contrast), clinical_question=COALESCE(NULLIF(?,''), clinical_question), meta_json=? WHERE id=?",
                    (
                        modality or "",
                        body_region or "",
                        contrast or "",
                        clinical_question or "",
                        json.dumps(study_meta or {}, ensure_ascii=False),
                        int(study_id),
                    ),
                )
                conn.commit()
        except Exception:
            pass

        _progress(100, "Готово")
        return IngestResult(
            study_id=int(study_id),
            n_previews=int(n_prev),
            modality=(modality or "").upper(),
            series_uid=selected_series_uid or (str(series_uid) if series_uid else ""),
        )


def _generate_previews_from_dicom_zip(
    db: Database,
    *,
    study_id: int,
    zip_path: str,
    series_uid: str | None = None,
    max_previews: int = 12,
    progress_cb: Optional[Callable[[int, str, str], None]] = None,
) -> tuple[int, str, dict[str, Any]]:
    zpath = Path(zip_path)
    if not zpath.exists():
        raise FileNotFoundError(str(zpath))

    extracted: dict[str, Any] = {}
    with zipfile.ZipFile(str(zpath), "r") as zf:
        if progress_cb is not None:
            try:
                progress_cb(15, "Сканирую DICOM-заголовки…", "")
            except Exception:
                pass

        series_index = _dicom_series_index_from_zip(zf)

        # Choose series
        chosen_sid = ""
        chosen_items: list[dict[str, Any]] = []
        if series_uid and str(series_uid) in series_index:
            chosen_sid = str(series_uid)
            chosen_items = series_index.get(chosen_sid, [])
        else:
            chosen_sid, chosen_items = _choose_best_series(series_index)

        if not chosen_sid or not chosen_items:
            raise RuntimeError("Не удалось найти DICOM-серию в ZIP (или файлы не распознаны как DICOM).")

        items = _sort_slices(chosen_items)

        # Pick indices evenly across the stack
        n = len(items)
        n_take = max(1, min(int(max_previews), n))
        idxs = np.linspace(0, n - 1, num=n_take, dtype=int).tolist()

        # Load one slice for meta/modality inference
        mid_name = items[idxs[len(idxs) // 2]]["name"]
        first_arr, first_meta = _load_dicom_pixels_from_zip(zf, mid_name)
        modality = (first_meta.get("Modality") or "").upper()

        extracted.update(
            {
                "series_uid": chosen_sid,
                "series_desc": items[0].get("series_desc", ""),
                "study_desc": items[0].get("study_desc", ""),
                "body_part": items[0].get("body", ""),
                "slices": n,
                "rows": first_meta.get("Rows", 0),
                "cols": first_meta.get("Columns", 0),
                "pixel_spacing": first_meta.get("PixelSpacing", []),
                "slice_thickness": first_meta.get("SliceThickness", ""),
            }
        )

        # Decide display transform
        if modality == "CT":
            wl, ww = 40.0, 400.0
        else:
            wl, ww = 0.0, 0.0

        out_count = 0
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            for k, idx in enumerate(idxs, start=1):
                if progress_cb is not None:
                    try:
                        pct = 20 + int((k / max(1, len(idxs))) * 75)
                        progress_cb(pct, f"Генерирую preview {k}/{len(idxs)}…", "")
                    except Exception:
                        pass

                name = items[idx]["name"]
                try:
                    arr, _meta = _load_dicom_pixels_from_zip(zf, name)
                except Exception:
                    continue

                if modality == "CT":
                    img = _to_uint8_windowed(arr, wl=wl, ww=ww)
                else:
                    img = _to_uint8_percentile(arr)

                out_png = td_p / f"dicom_axial_{k:02d}.png"
                try:
                    _save_png_uint8(img, out_png)
                except Exception:
                    continue

                stored = import_file(str(out_png), category="imaging_previews")
                db.imaging_add_preview(
                    study_id=study_id,
                    kind="axial",
                    label=f"Axial {k}/{len(idxs)}",
                    storage_path=stored.storage_path,
                )
                out_count += 1

                if progress_cb is not None:
                    try:
                        progress_cb(20 + int((k / max(1, len(idxs))) * 75), "Preview добавлен", stored.storage_path)
                    except Exception:
                        pass

        return out_count, modality, extracted


def _generate_previews_from_nifti(
    db: Database,
    *,
    study_id: int,
    nifti_path: str,
    max_previews: int = 12,
    progress_cb: Optional[Callable[[int, str, str], None]] = None,
) -> tuple[int, dict[str, Any]]:
    try:
        import nibabel as nib
    except Exception as e:
        raise RuntimeError("nibabel is required for NIfTI import. Please install requirements.") from e

    p = Path(nifti_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    img = nib.load(str(p))
    data = img.get_fdata(dtype=np.float32)

    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"Unsupported NIfTI shape: {data.shape}")

    z = data.shape[2]
    n_take = max(1, min(int(max_previews), z))
    idxs = np.linspace(0, z - 1, num=n_take, dtype=int).tolist()

    extracted: dict[str, Any] = {"format": "nifti", "shape": list(data.shape)}
    try:
        extracted["zooms"] = list(img.header.get_zooms()[:3])
    except Exception:
        pass

    out_count = 0
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for k, idx in enumerate(idxs, start=1):
            if progress_cb is not None:
                try:
                    pct = 20 + int((k / max(1, len(idxs))) * 75)
                    progress_cb(pct, f"Генерирую preview {k}/{len(idxs)}…", "")
                except Exception:
                    pass
            arr = data[:, :, idx]
            arr2 = np.rot90(arr)
            img8 = _to_uint8_percentile(arr2)
            out_png = td_p / f"nifti_axial_{k:02d}.png"
            _save_png_uint8(img8, out_png)
            stored = import_file(str(out_png), category="imaging_previews")
            db.imaging_add_preview(study_id=study_id, kind="axial", label=f"Axial {k}/{len(idxs)}", storage_path=stored.storage_path)
            out_count += 1
            if progress_cb is not None:
                try:
                    progress_cb(20 + int((k / max(1, len(idxs))) * 75), "Preview добавлен", stored.storage_path)
                except Exception:
                    pass

    return out_count, extracted


# -------------------- Context injection --------------------

def build_radiology_context(db: Database, dialog_id: int, *, max_previews: int = 12) -> tuple[str, list[str]]:
    """Return ([RADIOLOGY CONTEXT] block, preview_paths)."""
    st = None
    try:
        st = db.imaging_get_active(int(dialog_id))
    except Exception:
        st = None

    if not st:
        return "", []

    meta = st.get("meta") or {}
    previews = list(st.get("previews") or [])
    preview_paths = [p.get("storage_path") for p in previews if (p.get("storage_path") or "").strip()]
    if max_previews and len(preview_paths) > int(max_previews):
        preview_paths = preview_paths[: int(max_previews)]

    lines: list[str] = ["[RADIOLOGY CONTEXT]"]
    if st.get("modality"):
        lines.append(f"Modality: {st['modality']}")
    if st.get("body_region"):
        lines.append(f"Region: {st['body_region']}")
    if st.get("contrast"):
        lines.append(f"Contrast: {st['contrast']}")
    if st.get("clinical_question"):
        lines.append(f"Clinical question: {st['clinical_question']}")

    # Non-identifying technical metadata
    if meta.get("study_desc"):
        lines.append(f"Study: {meta.get('study_desc')}")
    if meta.get("series_desc"):
        lines.append(f"Series: {meta.get('series_desc')}")
    if meta.get("body_part"):
        lines.append(f"Body part (DICOM): {meta.get('body_part')}")
    if meta.get("series_uid"):
        lines.append(f"SeriesInstanceUID: {meta.get('series_uid')}")
    if meta.get("slices"):
        lines.append(f"Slices: {meta.get('slices')}")
    if meta.get("rows") and meta.get("cols"):
        lines.append(f"Matrix: {meta.get('rows')}x{meta.get('cols')}")
    if meta.get("pixel_spacing"):
        lines.append(f"Pixel spacing: {meta.get('pixel_spacing')}")
    if meta.get("slice_thickness"):
        lines.append(f"Slice thickness: {meta.get('slice_thickness')}")
    if meta.get("shape"):
        lines.append(f"Volume shape: {meta.get('shape')}")
    if meta.get("zooms"):
        lines.append(f"Voxel size: {meta.get('zooms')}")

    if preview_paths:
        lines.append(f"Previews attached: {len(preview_paths)} image(s) (axial samples).")
    else:
        lines.append("No previews available.")

    return "\n".join(lines).strip(), preview_paths


# -------------------- Viewer loading (draft MPR) --------------------

def load_volume_for_viewer(db: Database, study_id: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a 3D volume (int16) + meta for viewer.

    Supports DICOM ZIP (preferred) and NIfTI.

    NOTE: This is a *draft* viewer loader, meant for interactive slice viewing.
    """
    st = db.imaging_get_study(int(study_id))
    if not st:
        raise RuntimeError("Study not found")

    meta = st.get("meta") or {}
    source_path = str(st.get("source_storage_path") or "")
    source_type = str(st.get("source_type") or "")

    if source_type in {"dicom_zip", "dicom_dir"} and source_path.lower().endswith(".zip"):
        series_uid = str(meta.get("series_uid") or meta.get("preferred_series_uid") or "")
        vol, meta2 = _load_dicom_volume_from_zip(source_path, series_uid=series_uid)
        meta.update(meta2)
        return vol, meta

    if source_type == "nifti" and (source_path.lower().endswith(".nii") or source_path.lower().endswith(".nii.gz")):
        vol, meta2 = _load_nifti_volume(source_path)
        meta.update(meta2)
        return vol, meta

    raise RuntimeError("Viewer supports only DICOM ZIP or NIfTI studies")


def _load_dicom_volume_from_zip(zip_path: str, *, series_uid: str = "") -> tuple[np.ndarray, dict[str, Any]]:
    """Load selected series from ZIP into (Z,H,W) int16 volume."""
    try:
        import pydicom
    except Exception as e:
        raise RuntimeError("pydicom is required for DICOM viewer.") from e

    zpath = Path(zip_path)
    if not zpath.exists():
        raise FileNotFoundError(str(zpath))

    with zipfile.ZipFile(str(zpath), "r") as zf:
        series_index = _dicom_series_index_from_zip(zf)
        chosen_sid = ""
        chosen_items: list[dict[str, Any]] = []
        if series_uid and series_uid in series_index:
            chosen_sid = series_uid
            chosen_items = series_index.get(chosen_sid, [])
        else:
            chosen_sid, chosen_items = _choose_best_series(series_index)

        if not chosen_sid or not chosen_items:
            raise RuntimeError("No DICOM series found")

        items = _sort_slices(chosen_items)

        modality = ""
        slope = 1.0
        intercept = 0.0
        slices: list[np.ndarray] = []

        # Load first to get modality + rescale
        data0 = zf.read(items[0]["name"])
        ds0 = pydicom.dcmread(pydicom.filebase.DicomBytesIO(data0), force=True)
        modality = _safe_str(getattr(ds0, "Modality", "") or "").upper()
        slope = float(getattr(ds0, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds0, "RescaleIntercept", 0.0) or 0.0)

        for it in items:
            data = zf.read(it["name"])
            ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(data), force=True)
            arr = ds.pixel_array
            a = arr.astype(np.float32)
            if modality == "CT":
                a = a * slope + intercept
            # HU and typical MR intensities fit int16; clip to be safe
            a = np.clip(a, -32768, 32767)
            slices.append(a.astype(np.int16))

        vol = np.stack(slices, axis=0)  # (Z,H,W)

        meta = {
            "series_uid": chosen_sid,
            "modality": modality,
            "rows": int(vol.shape[1]),
            "cols": int(vol.shape[2]),
            "slices": int(vol.shape[0]),
            "rescale_slope": slope,
            "rescale_intercept": intercept,
        }
        return vol, meta


def _load_nifti_volume(nifti_path: str) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import nibabel as nib
    except Exception as e:
        raise RuntimeError("nibabel is required for NIfTI viewer.") from e

    p = Path(nifti_path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    img = nib.load(str(p))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"Unsupported NIfTI shape: {data.shape}")

    # We'll store as int16 for memory.
    a = np.clip(data, -32768, 32767).astype(np.int16)
    # NIfTI order is often (X,Y,Z) - we map to (Z,Y,X)
    vol = np.transpose(a, (2, 1, 0))

    meta: dict[str, Any] = {"format": "nifti", "shape": list(data.shape)}
    try:
        meta["zooms"] = list(img.header.get_zooms()[:3])
    except Exception:
        pass

    return vol, meta
