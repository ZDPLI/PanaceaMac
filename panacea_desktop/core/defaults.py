from __future__ import annotations

"""Default settings for the desktop app.

These defaults are intentionally conservative and OpenAI-compatible.
All of them can be changed in the Settings dialog.
"""

DEFAULTS: dict[str, str] = {
    # Provider routing
    "provider_mode": "auto",  # auto|novita|openrouter|custom
    "novita_base_url": "https://api.novita.ai/openai",
    "openrouter_base_url": "https://openrouter.ai/api/v1",
    "custom_base_url": "",

    # API keys (prefer environment variables)
    "novita_api_key": "",
    "openrouter_api_key": "",
    "custom_api_key": "",

    # Model routing
    # Provider-specific model ids. These can be overridden in Settings and/or via env vars.
    # Novita model ids
    "novita_model_light": "qwen/qwen2.5-vl-72b-instruct",
    "novita_model_medium": "baidu/ernie-4.5-vl-424b-a47b",
    "novita_model_heavy": "zai-org/glm-4.5v",
    "novita_model_arbiter": "openai/gpt-oss-120b",

    # OpenRouter model ids
    "openrouter_model_light": "qwen/qwen3-vl-32b-instruct",
    "openrouter_model_medium": "baidu/ernie-4.5-vl-424b-a47b",
    "openrouter_model_heavy": "z-ai/glm-4.6v",
    "openrouter_model_arbiter": "openai/gpt-oss-120b",

    # Custom provider (generic)
    "model_light": "qwen/qwen3-vl-32b-instruct",
    "model_medium": "baidu/ernie-4.5-vl-424b-a47b",
    "model_heavy": "z-ai/glm-4.6v",
    "model_arbiter": "openai/gpt-oss-120b",

    # Generation params
    "temperature": "0.2",
    "max_tokens": "1200",
    "history_max_messages": "30",

    # Global cross-dialog memory
    "global_memory_max_items": "12",
    "auto_memory_enabled": "0",


    # Cross-turn memory toggle (UI switch). If 0, the engine will not include
    # previous turns in the prompt (single-turn mode).
    "memory_enabled": "1",

    # Modes
    "prompt_pack": "gp",  # gp|derm|histo|radio|daughter
    "reasoning_mode": "medium",  # light|medium|heavy|consensus

    # RAG
    "rag_enabled": "1",  # 1|0
    "rag_top_k": "6",
    "rag_chunk_chars": "1200",
    "rag_chunk_overlap_chars": "200",
    "rag_retrieval_mode": "bm25",  # lexical|bm25|faiss
    "rag_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "rag_index_dirty": "1",

    # UI
    "active_dialog_id": "",
}

# Environment variable names (optional)
ENV_MAP: dict[str, str] = {
    "novita_api_key": "NOVITA_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "custom_api_key": "PANACEA_API_KEY",
    "custom_base_url": "PANACEA_BASE_URL",

    # Optional base URLs
    "novita_base_url": "NOVITA_BASE_URL",
    "openrouter_base_url": "OPENROUTER_BASE_URL",

    # Optional model ids
    "novita_model_light": "NOVITA_MODEL_LIGHT",
    "novita_model_medium": "NOVITA_MODEL_MEDIUM",
    "novita_model_heavy": "NOVITA_MODEL_HEAVY",
    "novita_model_arbiter": "NOVITA_MODEL_ARBITER",

    # OpenRouter model ids can be overridden via generic env vars
    "openrouter_model_light": "MODEL_LIGHT",
    "openrouter_model_medium": "MODEL_MEDIUM",
    "openrouter_model_heavy": "MODEL_HEAVY",
    "openrouter_model_arbiter": "MODEL_ARBITER",

}
