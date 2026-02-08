from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .db import Database
from .llm_client import LLMError, OpenAICompatibleClient, ProviderConfig
from .prompts import PROMPT_PACKS
from .rag import build_rag_context
from .radiology import build_radiology_context


# Some upstream LLM providers return boilerplate refusals for radiology-image questions.
# We keep responses useful by retrying once with a stronger non-refusal instruction.
_REFUSAL_RE = re.compile(
    r"(?is)\b(к\s*сожалению\b.*?не\s*могу|не\s*могу\s+предостав(ить|лять)\s+медицинск|"
    r"только\s+квалифицированн(ый|ая)\s+специалист|обратит(есь|ьтесь)\s+к\s+(врачу|радиологу)|"
    r"не\s+имею\s+права\s+интерпретир|не\s+могу\s+интерпретир)\b"
)


def looks_like_refusal(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return bool(_REFUSAL_RE.search(t))


ANTI_REFUSAL_SYSTEM = (
    "[ANTI-REFUSAL]\n"
    "Вы отвечаете полезно и по существу даже для рентгенологических данных. "
    "Не используйте полный отказ ('не могу', 'не имею права', 'только врач может'). "
    "Вместо отказа: дайте ориентировочное описание видимых признаков на предоставленных предпросмотрах, "
    "подчеркните, что это НЕ окончательное заключение без полного DICOM/клиники, "
    "и предложите безопасные следующие шаги (что уточнить, какие серии/параметры/сравнение)."
)


@dataclass
class EngineResult:
    text: str
    provider_used: str
    model_used: str
    candidates: dict[str, str] | None = None


def _read_setting(db: Database, key: str, cast=str, default=None):
    v = db.get_setting(key, default=None)
    if v is None:
        return default
    try:
        return cast(v)
    except Exception:
        return default


def _image_to_data_url(path: str) -> str:
    p = Path(path)
    data = p.read_bytes()
    ext = p.suffix.lower().lstrip(".") or "png"
    if ext == "jpg":
        ext = "jpeg"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/{ext};base64,{b64}"


class PanaceaEngine:
    def __init__(self, db: Database):
        self.db = db

    # ---------- provider resolution ----------
    def _provider_configs(self) -> dict[str, ProviderConfig]:
        novita = ProviderConfig(
            name="novita",
            base_url=_read_setting(self.db, "novita_base_url", str, "https://api.novita.ai/openai"),
            api_key=_read_setting(self.db, "novita_api_key", str, ""),
            extra_headers={},
        )
        openrouter = ProviderConfig(
            name="openrouter",
            base_url=_read_setting(self.db, "openrouter_base_url", str, "https://openrouter.ai/api/v1"),
            api_key=_read_setting(self.db, "openrouter_api_key", str, ""),
            extra_headers={"HTTP-Referer": "", "X-Title": "PanaceaDesktop"},
        )
        custom = ProviderConfig(
            name="custom",
            base_url=_read_setting(self.db, "custom_base_url", str, ""),
            api_key=_read_setting(self.db, "custom_api_key", str, ""),
            extra_headers={},
        )
        return {"novita": novita, "openrouter": openrouter, "custom": custom}

    def _provider_order(self) -> list[str]:
        mode = (_read_setting(self.db, "provider_mode", str, "auto") or "auto").lower().strip()
        if mode in {"novita", "openrouter", "custom"}:
            return [mode]
        order = ["novita", "openrouter"]
        if (_read_setting(self.db, "custom_base_url", str, "") or "").strip():
            order.append("custom")
        return order

    def _provider_order_usable(self) -> list[ProviderConfig]:
        configs = self._provider_configs()
        usable: list[ProviderConfig] = []
        for name in self._provider_order():
            cfg = configs.get(name)
            if not cfg:
                continue
            if not (cfg.base_url or "").strip():
                continue
            if not (cfg.api_key or "").strip():
                continue
            usable.append(cfg)
        return usable

    # ---------- voice (STT) ----------
    def transcribe_audio(self, file_path: str) -> tuple[str, str]:
        """Transcribe an audio file using an OpenAI-compatible /audio/transcriptions endpoint.

        Returns (text, provider_name) on success.
        """
        stt_model = (_read_setting(self.db, "voice_stt_model", str, "whisper-1") or "whisper-1").strip()
        language = (_read_setting(self.db, "voice_stt_language", str, "") or "").strip() or None
        prompt = (_read_setting(self.db, "voice_stt_prompt", str, "") or "").strip() or None

        last_err: Exception | None = None
        for cfg in self._provider_order_usable():
            try:
                client = OpenAICompatibleClient(cfg)
                text = client.audio_transcriptions(model=stt_model, file_path=file_path, language=language, prompt=prompt)
                text = (text or "").strip()
                if text:
                    return text, cfg.name
                return "", cfg.name
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        raise LLMError("No usable provider configured for voice transcription")

    # ---------- prompts / message assembly ----------
    def _system_prompt(self) -> str:
        pack = _read_setting(self.db, "prompt_pack", str, "gp")
        pack = pack if pack in PROMPT_PACKS else "gp"
        ov = self.db.get_prompt_override(pack)
        if ov and (ov.get("system_prompt") or "").strip():
            return ov["system_prompt"]
        return PROMPT_PACKS[pack]["system"]

    def _arbiter_system_prompt(self) -> str:
        pack = _read_setting(self.db, "prompt_pack", str, "gp")
        pack = pack if pack in PROMPT_PACKS else "gp"
        ov = self.db.get_prompt_override(pack)
        if ov and (ov.get("arbiter_prompt") or "").strip():
            return ov["arbiter_prompt"]
        return PROMPT_PACKS[pack]["arbiter"]

    def _histo_block(self, dialog_id: int) -> str:
        pack = _read_setting(self.db, "prompt_pack", str, "gp")
        if pack != "histo":
            return ""
        st = self.db.histo_get(dialog_id)
        if not any(st.values()):
            return ""
        parts = ["[HISTO METADATA]"]
        if st.get("stain"):
            parts.append(f"Окраска: {st['stain']}")
        if st.get("magnification"):
            parts.append(f"Увеличение: {st['magnification']}")
        if st.get("quality"):
            parts.append(f"Качество: {st['quality']}")
        if st.get("note"):
            parts.append(f"Комментарий: {st['note']}")
        return "\n".join(parts).strip()

    def _build_messages(
        self, dialog_id: int, user_text: str, image_paths: list[str], *, extra_system: str | None = None
    ) -> list[dict[str, Any]]:
        memory_enabled = _read_setting(self.db, "memory_enabled", int, 1) == 1
        history_max = _read_setting(self.db, "history_max_messages", int, 30)

        msgs = self.db.get_messages(dialog_id) if memory_enabled else []
        if msgs and history_max and len(msgs) > history_max:
            msgs = msgs[-history_max:]

        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        if extra_system and (extra_system or "").strip():
            messages.append({"role": "system", "content": (extra_system or "").strip()})

        rag_enabled = _read_setting(self.db, "rag_enabled", int, 1) == 1
        if rag_enabled:
            top_k = _read_setting(self.db, "rag_top_k", int, 6)
            rag_ctx = build_rag_context(self.db, user_text, top_k=top_k, dialog_id=dialog_id)
            if rag_ctx:
                messages.append({"role": "system", "content": f"[RAG CONTEXT]\n{rag_ctx}"})

        histo = self._histo_block(dialog_id)
        if histo:
            messages.append({"role": "system", "content": histo})

        # Radiology / imaging context (CT/MRI previews + metadata)
        radio_ctx, radio_images = build_radiology_context(self.db, dialog_id, max_previews=12)
        if radio_ctx:
            messages.append({"role": "system", "content": radio_ctx})

        # Cross-dialog global memory
        if memory_enabled:
            mem_limit = _read_setting(self.db, "global_memory_max_items", int, 12)
            try:
                mem_items = self.db.list_global_memories(
                    include_disabled=False, limit=int(mem_limit) if mem_limit else None
                )
            except Exception:
                mem_items = []
            if mem_items:
                mem_lines: list[str] = []
                for mi in mem_items:
                    txt = (mi.get("content") or "").strip()
                    if not txt:
                        continue
                    prefix = "📌 " if int(mi.get("pinned") or 0) == 1 else "• "
                    mem_lines.append(prefix + txt)
                if mem_lines:
                    messages.append({"role": "system", "content": "[MIRIAM MEMORY]\n" + "\n".join(mem_lines)})

        for m in msgs:
            role = m.get("role")
            content = m.get("content") or ""
            if role not in {"user", "assistant"}:
                continue
            atts = m.get("attachments") or {}
            if atts.get("images"):
                content += "\n\n[Вложение: изображение(я) присутствуют в оригинальном чате.]"
            messages.append({"role": role, "content": content})

        merged_images = []
        # Put radiology previews first (if any), then user-attached images for this turn.
        if radio_images:
            merged_images.extend(radio_images)
        if image_paths:
            merged_images.extend(image_paths)

        if merged_images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            for p in merged_images:
                try:
                    url = _image_to_data_url(p)
                except Exception:
                    continue
                parts.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": parts})
        else:
            messages.append({"role": "user", "content": user_text})

        return messages

    # ---------- model selection ----------
    def _models_by_provider(self) -> dict[str, dict[str, str]]:
        novita = {
            "light": _read_setting(self.db, "novita_model_light", str, "qwen/qwen2.5-vl-72b-instruct"),
            "medium": _read_setting(self.db, "novita_model_medium", str, "baidu/ernie-4.5-vl-424b-a47b"),
            "heavy": _read_setting(self.db, "novita_model_heavy", str, "zai-org/glm-4.5v"),
            "arbiter": _read_setting(self.db, "novita_model_arbiter", str, "openai/gpt-oss-120b"),
        }
        openrouter = {
            "light": _read_setting(self.db, "openrouter_model_light", str, "qwen/qwen3-vl-32b-instruct"),
            "medium": _read_setting(self.db, "openrouter_model_medium", str, "baidu/ernie-4.5-vl-424b-a47b"),
            "heavy": _read_setting(self.db, "openrouter_model_heavy", str, "z-ai/glm-4.6v"),
            "arbiter": _read_setting(self.db, "openrouter_model_arbiter", str, "openai/gpt-oss-120b"),
        }
        custom = {
            "light": _read_setting(self.db, "model_light", str, "qwen/qwen3-vl-32b-instruct"),
            "medium": _read_setting(self.db, "model_medium", str, "baidu/ernie-4.5-vl-424b-a47b"),
            "heavy": _read_setting(self.db, "model_heavy", str, "z-ai/glm-4.6v"),
            "arbiter": _read_setting(self.db, "model_arbiter", str, "openai/gpt-oss-120b"),
        }
        return {"novita": novita, "openrouter": openrouter, "custom": custom}

    def _model_map(self, tier: str) -> dict[str, str]:
        tier = (tier or "medium").lower().strip()
        if tier not in {"light", "medium", "heavy", "arbiter"}:
            tier = "medium"
        models = self._models_by_provider()
        return {p: mp.get(tier, mp.get("medium", "")) for p, mp in models.items()}

    def _model_fallbacks(self, requested: str) -> list[str]:
        requested = (requested or "").strip()
        if not requested:
            return []
        if "/" in requested and not requested.startswith("gpt-"):
            return [requested]

        candidates = [
            requested,
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4.1",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]
        out: list[str] = []
        for m in candidates:
            m = (m or "").strip()
            if m and m not in out:
                out.append(m)
        return out

    # ---------- provider calling ----------
    def _call_first_available(
        self, *, model: str | dict[str, str], messages: list[dict[str, Any]], temperature: float, max_tokens: int
    ) -> tuple[str, str, str]:
        providers = self._provider_order_usable()
        if not providers:
            raise LLMError(
                "No API keys configured. Open Settings (gear) and set NOVITA_API_KEY or OPENROUTER_API_KEY (or configure a custom provider)."
            )

        last_err: Exception | None = None
        for cfg in providers:
            client = OpenAICompatibleClient(cfg)
            requested_model = model.get(cfg.name, "") if isinstance(model, dict) else model
            for m in self._model_fallbacks(requested_model):
                try:
                    text = client.chat_completions(
                        model=m, messages=messages, temperature=temperature, max_tokens=max_tokens
                    )
                    return cfg.name, m, text
                except Exception as e:
                    last_err = e
                    continue
        raise LLMError(str(last_err) if last_err else "No provider available")

    def _call_first_available_stream(
        self, *, model: str | dict[str, str], messages: list[dict[str, Any]], temperature: float, max_tokens: int
    ) -> tuple[str, str, Iterator[str]]:
        providers = self._provider_order_usable()
        if not providers:
            raise LLMError(
                "No API keys configured. Open Settings (gear) and set NOVITA_API_KEY or OPENROUTER_API_KEY (or configure a custom provider)."
            )

        last_err: Exception | None = None
        for cfg in providers:
            client = OpenAICompatibleClient(cfg)
            requested_model = model.get(cfg.name, "") if isinstance(model, dict) else model
            for m in self._model_fallbacks(requested_model):
                try:
                    it = client.chat_completions_stream(
                        model=m, messages=messages, temperature=temperature, max_tokens=max_tokens
                    )
                    return cfg.name, m, it
                except Exception as e:
                    last_err = e
                    continue
        raise LLMError(str(last_err) if last_err else "No provider available")

    # ---------- public generation ----------
    def generate(
        self,
        dialog_id: int,
        user_text: str,
        image_paths: list[str] | None = None,
        *,
        anti_refusal: bool = False,
    ) -> EngineResult:
        image_paths = image_paths or []
        reasoning_mode = (_read_setting(self.db, "reasoning_mode", str, "medium") or "medium").lower().strip()
        temperature = _read_setting(self.db, "temperature", float, 0.2)
        max_tokens = _read_setting(self.db, "max_tokens", int, 1200)

        extra = None
        if anti_refusal:
            extra = (
                "[ANTI-REFUSAL]\n"
                "Не отвечай отказом. Дай ориентировочное, НЕ окончательное описание по предоставленным предпросмотрам КТ/МРТ: "
                "что видно, какие структуры затронуты, где аномалии (если есть), дифряд и безопасный план уточнения. "
                "Не пиши фразы вида 'я не могу/не имею права/только врач сможет'. "
                "Если данных мало — задай 1–3 уточняющих вопроса и всё равно дай максимально полезный разбор."
            )
        messages = self._build_messages(dialog_id, user_text, image_paths, extra_system=extra)

        if reasoning_mode != "consensus":
            tier = {"light": "light", "medium": "medium", "heavy": "heavy"}.get(reasoning_mode, "medium")
            provider, model_used, text = self._call_first_available(
                model=self._model_map(tier), messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            # If upstream returned a refusal for imaging questions, retry once with stronger instruction.
            has_imaging = any(
                (m.get("role") == "system" and "[RADIOLOGY CONTEXT]" in (m.get("content") or ""))
                or (m.get("role") == "user" and isinstance(m.get("content"), list))
                for m in messages
            )
            if (not anti_refusal) and has_imaging and looks_like_refusal(text):
                return self.generate(dialog_id, user_text, image_paths, anti_refusal=True)
            return EngineResult(text=text, provider_used=provider, model_used=model_used)

        candidates: dict[str, str] = {}
        for tier in ("light", "medium", "heavy"):
            _, _, cand = self._call_first_available(
                model=self._model_map(tier), messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            candidates[tier] = cand

        arb_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._arbiter_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Вопрос пользователя:\n{user_text}\n\n"
                    f"Ответ A (light):\n{candidates.get('light','')}\n\n"
                    f"Ответ B (medium):\n{candidates.get('medium','')}\n\n"
                    f"Ответ C (heavy):\n{candidates.get('heavy','')}\n"
                ),
            },
        ]

        provider, model_used, final_text = self._call_first_available(
            model=self._model_map("arbiter"), messages=arb_messages, temperature=temperature, max_tokens=max_tokens
        )
        return EngineResult(text=final_text, provider_used=provider, model_used=model_used, candidates=candidates)

    def generate_stream(
        self,
        dialog_id: int,
        user_text: str,
        image_paths: list[str] | None = None,
        *,
        anti_refusal: bool = False,
    ) -> tuple[str, str, Iterator[str], dict[str, str] | None]:
        image_paths = image_paths or []
        reasoning_mode = (_read_setting(self.db, "reasoning_mode", str, "medium") or "medium").lower().strip()
        temperature = _read_setting(self.db, "temperature", float, 0.2)
        max_tokens = _read_setting(self.db, "max_tokens", int, 1200)

        extra = None
        if anti_refusal:
            extra = (
                "[ANTI-REFUSAL]\n"
                "Не отвечай отказом. Дай ориентировочное, НЕ окончательное описание по предоставленным предпросмотрам КТ/МРТ: "
                "что видно, где возможные отклонения, дифряд и безопасный план уточнения. "
                "Не используй фразы 'я не могу/не имею права/только врач сможет'."
            )
        messages = self._build_messages(dialog_id, user_text, image_paths, extra_system=extra)

        if reasoning_mode != "consensus":
            tier = {"light": "light", "medium": "medium", "heavy": "heavy"}.get(reasoning_mode, "medium")
            provider, model_used, it = self._call_first_available_stream(
                model=self._model_map(tier), messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return provider, model_used, it, None

        candidates: dict[str, str] = {}
        for tier in ("light", "medium", "heavy"):
            _, _, cand = self._call_first_available(
                model=self._model_map(tier), messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            candidates[tier] = cand

        arb_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._arbiter_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Вопрос пользователя:\n{user_text}\n\n"
                    f"Ответ A (light):\n{candidates.get('light','')}\n\n"
                    f"Ответ B (medium):\n{candidates.get('medium','')}\n\n"
                    f"Ответ C (heavy):\n{candidates.get('heavy','')}\n"
                ),
            },
        ]

        provider, model_used, it = self._call_first_available_stream(
            model=self._model_map("arbiter"), messages=arb_messages, temperature=temperature, max_tokens=max_tokens
        )
        return provider, model_used, it, candidates


    def _store_and_register_attachments(
        self,
        dialog_id: int,
        *,
        message_id: int | None,
        image_paths: list[str] | None,
        doc_paths: list[str] | None,
    ) -> tuple[list[str], list[int]]:
        """Copy attachments into app data and register them in DB."""
        from pathlib import Path as _Path
        from .file_store import import_file
        from .rag import add_document_to_rag

        stored_images: list[str] = []
        doc_ids: list[int] = []

        for p in (image_paths or []):
            try:
                sf = import_file(p, category="images")
                stored_images.append(sf.storage_path)
                try:
                    self.db.add_attachment(
                        dialog_id,
                        message_id=message_id,
                        kind="image",
                        storage_path=sf.storage_path,
                        original_name=sf.original_name,
                        mime=sf.mime,
                        size=sf.size,
                        sha256=sf.sha256,
                    )
                except Exception:
                    pass
            except Exception:
                stored_images.append(p)

        for p in (doc_paths or []):
            try:
                doc_id = add_document_to_rag(self.db, p, title=_Path(p).name, scope="dialog", dialog_id=dialog_id)
                doc_ids.append(int(doc_id))
            except Exception:
                pass
            try:
                sf = import_file(p, category="docs")
                try:
                    self.db.add_attachment(
                        dialog_id,
                        message_id=message_id,
                        kind="doc",
                        storage_path=sf.storage_path,
                        original_name=sf.original_name,
                        mime=sf.mime,
                        size=sf.size,
                        sha256=sf.sha256,
                    )
                except Exception:
                    pass
            except Exception:
                pass

        return stored_images, doc_ids

    # ---------- persistence helpers ----------

    def handle_user_turn(
        self,
        dialog_id: int,
        user_text: str,
        image_paths: list[str] | None = None,
        doc_paths: list[str] | None = None,
    ) -> EngineResult:
        image_paths = image_paths or []
        doc_paths = doc_paths or []
        user_msg_id = self.db.add_message(dialog_id, "user", user_text, attachments={})
        stored_images, _doc_ids = self._store_and_register_attachments(
            dialog_id, message_id=user_msg_id, image_paths=image_paths, doc_paths=doc_paths
        )
        attachments = {}
        if stored_images:
            attachments["images"] = stored_images
        if doc_paths:
            # keep originals for UI; documents are copied into app storage separately
            attachments["docs"] = doc_paths
        if attachments:
            try:
                self.db.set_message_attachments(user_msg_id, attachments)
            except Exception:
                pass

        result = self.generate(dialog_id, user_text, stored_images)
        self.db.add_message(dialog_id, "assistant", result.text, attachments={})
        return result



    def handle_user_turn_stream(
        self,
        dialog_id: int,
        user_text: str,
        image_paths: list[str] | None = None,
        doc_paths: list[str] | None = None,
    ) -> tuple[int, str, str, Iterator[str], dict[str, str] | None]:
        image_paths = image_paths or []
        doc_paths = doc_paths or []
        user_msg_id = self.db.add_message(dialog_id, "user", user_text, attachments={})
        stored_images, _doc_ids = self._store_and_register_attachments(
            dialog_id, message_id=user_msg_id, image_paths=image_paths, doc_paths=doc_paths
        )
        attachments = {}
        if stored_images:
            attachments["images"] = stored_images
        if doc_paths:
            attachments["docs"] = doc_paths
        if attachments:
            try:
                self.db.set_message_attachments(user_msg_id, attachments)
            except Exception:
                pass

        assistant_msg_id = self.db.add_message(dialog_id, "assistant", "", attachments={})
        provider, model, it, candidates = self.generate_stream(dialog_id, user_text, stored_images)
        return assistant_msg_id, provider, model, it, candidates



    # ---------- auto memory extraction ----------
    def maybe_extract_and_store_memory(self, dialog_id: int, user_text: str, assistant_text: str) -> list[str]:
        """Optionally extract stable cross-dialog memories and store them.

        Controlled by settings:
        - auto_memory_enabled (0/1)
        - global_memory_max_items (used for injection)
        """
        enabled = _read_setting(self.db, "auto_memory_enabled", int, 0) == 1
        if not enabled:
            return []
        # Use a light model tier; do NOT use consensus here.
        temperature = 0.0
        max_tokens = 300

        system = (
            "You are Miriam's memory module. Extract ONLY durable, user-approved, high-value memories "
            "(stable preferences, long-term facts, ongoing projects/constraints). "
            "Do NOT store sensitive medical data unless explicitly asked. "
            "Return STRICT JSON: an array of short strings. No prose."
        )
        prompt = (
            "Conversation snippet:\n"
            f"USER: {user_text}\n"
            f"ASSISTANT: {assistant_text}\n\n"
            "JSON array of memories:"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        provider, model_used, text = self._call_first_available(
            model=self._model_map("light"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Parse JSON robustly
        items: list[str] = []
        try:
            data = json.loads(text.strip())
            if isinstance(data, list):
                for x in data:
                    if isinstance(x, str) and x.strip():
                        items.append(x.strip())
        except Exception:
            # Fallback: try to find a JSON array in the text
            m = re.search(r"\[[\s\S]*\]", text)
            if m:
                try:
                    data = json.loads(m.group(0))
                    if isinstance(data, list):
                        for x in data:
                            if isinstance(x, str) and x.strip():
                                items.append(x.strip())
                except Exception:
                    pass

        # Store (deduplicate by exact match)
        existing = { (m.get("content") or "").strip() for m in self.db.list_global_memories(include_disabled=True) }
        stored: list[str] = []
        for it in items[:5]:
            if it in existing:
                continue
            try:
                self.db.add_global_memory(it, source="auto", dialog_id=dialog_id)
                stored.append(it)
                existing.add(it)
            except Exception:
                continue
        return stored


# Backwards/branding alias
MiriamEngine = PanaceaEngine
