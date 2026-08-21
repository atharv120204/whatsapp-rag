"""Central configuration, read from environment with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BACKEND_DIR / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # --- Paths ---------------------------------------------------------
    # Per-archive paths live on the Archive object; only device-wide locations
    # are here. Override DATA_DIR to keep archives somewhere else entirely.
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "").strip()
                                     or PROJECT_DIR / "data")
    )

    # --- Chat provider ---------------------------------------------------
    # The agent can talk to any endpoint speaking OpenAI chat-completions
    # (groq, xai, openrouter, ollama, openai, custom) or to Gemini natively.
    # Kept separate from the Gemini block below because embeddings and media
    # understanding stay on Gemini regardless -- most OpenAI-compatible hosts
    # serve no embeddings endpoint at all.
    chat_provider: str = field(default_factory=lambda: os.getenv("CHAT_PROVIDER", "gemini").strip().lower())
    chat_base_url: str = field(default_factory=lambda: os.getenv("CHAT_BASE_URL", "").strip())
    chat_api_key: str = field(default_factory=lambda: os.getenv("CHAT_API_KEY", "").strip())

    # --- Speech to text ---------------------------------------------------
    # Voice notes are usually the biggest group of attachments, and Groq serves
    # Whisper on a far more generous free tier than Gemini's. Moving them off
    # Gemini leaves its small daily allowance for images, which nothing else
    # here can read. Empty or "gemini" means Gemini handles audio as before.
    speech_provider: str = field(default_factory=lambda: os.getenv("SPEECH_PROVIDER", "").strip().lower())
    speech_model: str = field(default_factory=lambda: os.getenv("SPEECH_MODEL", "whisper-large-v3").strip())
    speech_base_url: str = field(default_factory=lambda: os.getenv("SPEECH_BASE_URL", "").strip())
    speech_api_key: str = field(default_factory=lambda: os.getenv("SPEECH_API_KEY", "").strip())
    # ISO-639-1 hint ("hi", "en", "es"). Empty means auto-detect. Whisper's
    # detection is unreliable on the short, noisy, code-mixed clips typical of
    # a voice note -- it read a few seconds of Hindi as Tagalog -- so naming
    # the language the group actually speaks measurably improves accuracy.
    speech_language: str = field(default_factory=lambda: os.getenv("SPEECH_LANGUAGE", "").strip())

    # --- Gemini --------------------------------------------------------
    # Model ids are configurable because Google ships new ones often; pinning
    # them in code means the app breaks on a rename it did not need to care
    # about. Run `python -m app.cli models` to list what your key can see.
    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    chat_model: str = field(default_factory=lambda: os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"))
    heavy_model: str = field(default_factory=lambda: os.getenv("GEMINI_HEAVY_MODEL", "gemini-2.5-pro"))
    vision_model: str = field(default_factory=lambda: os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash"))
    embed_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"))
    embed_dims: int = field(default_factory=lambda: _int("EMBED_DIMS", 768))

    # --- Ingest behaviour ----------------------------------------------
    session_gap_hours: float = field(default_factory=lambda: _float("SESSION_GAP_HOURS", 4.0))
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 25))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 5))
    embed_batch_size: int = field(default_factory=lambda: _int("EMBED_BATCH_SIZE", 64))

    # --- Media understanding -------------------------------------------
    describe_media: bool = field(default_factory=lambda: _bool("DESCRIBE_MEDIA", True))
    media_concurrency: int = field(default_factory=lambda: _int("MEDIA_CONCURRENCY", 4))
    max_video_mb: int = field(default_factory=lambda: _int("MAX_VIDEO_MB", 200))
    transcribe_audio: bool = field(default_factory=lambda: _bool("TRANSCRIBE_AUDIO", True))

    # --- Free-tier throttling -------------------------------------------
    # Google publishes free-tier limits per project in AI Studio rather than in
    # the docs, so these are a ceiling you set to match what your account
    # actually allows. 0 disables a limit.
    #
    # This used to default to 20/day, the observed free-tier allowance for
    # gemini-2.5-flash ("GenerateRequestsPerDayPerProjectPerModel-FreeTier,
    # limit: 20"), on the theory that a pessimistic guess is cheap because the
    # limiter adopts the real number from a 429. That theory only holds in one
    # direction. The limiter learns a *lower* limit by being refused, but it can
    # only learn a higher one by reaching the real ceiling -- which a cap set
    # below that ceiling prevents. So a low guess is self-reinforcing, not
    # self-correcting: it silently ends runs early and never discovers it had
    # room. It also conflates models whose real limits differ by two orders of
    # magnitude (flash vision is genuinely ~20/day; gemini-embedding-001 allows
    # far more, and one semantic search costs one request).
    #
    # 200 is therefore a stop-loss against a runaway agent loop rather than an
    # attempt to guess any model's quota. Whichever real per-day limit is hit
    # first still gets read out of the 429 and adopted for the rest of the day.
    max_requests_per_minute: int = field(
        default_factory=lambda: _int("MAX_REQUESTS_PER_MINUTE", 10))
    max_requests_per_day: int = field(
        default_factory=lambda: _int("MAX_REQUESTS_PER_DAY", 200))

    # --- Retrieval ------------------------------------------------------
    top_k: int = field(default_factory=lambda: _int("TOP_K", 12))
    max_sql_rows: int = field(default_factory=lambda: _int("MAX_SQL_ROWS", 500))
    # Steps are the dominant cost of a slow answer, and not linearly: every
    # step re-sends the whole conversation, so step 7 pays for the results of
    # steps 1-6. Measured against Groq's 8,000 tokens/minute, one question at
    # eight steps came to ~45,000 tokens -- four minutes of the wall clock spent
    # waiting out 429s, for an answer no better than the three-step version.
    # Four is enough for the shapes that actually occur: SQL, or search then
    # get_context, or SQL plus an illustrative search.
    # Two budgets, because one number cannot serve both shapes of question.
    # "How many messages did X send" is a single SQL call and should not be
    # given room to wander; "what did we decide about the trip" has to search,
    # read context, and weigh what it found, and cutting it off mid-way returns
    # nothing at all. The router picks per question -- see agent/router.py.
    agent_max_steps: int = field(default_factory=lambda: _int("AGENT_MAX_STEPS", 4))
    agent_max_steps_direct: int = field(
        default_factory=lambda: _int("AGENT_MAX_STEPS_DIRECT", 3))
    agent_max_steps_reasoning: int = field(
        default_factory=lambda: _int("AGENT_MAX_STEPS_REASONING", 8))

    # Free tiers cap tokens per minute, not just requests. This budget covers
    # the conversation only -- the system prompt and tool schema are fixed
    # overhead (~2900 tokens) and are excluded, so the number means the same
    # thing regardless of archive size. JSON tokenises at roughly three
    # characters per token. The agent halves this automatically if a request is
    # still refused as too large.
    #
    # Was 8000 characters, which fit inside Groq's 8,000 TPM for *one* request
    # and not for a conversation: at ~2700 tokens of results on top of ~2900
    # fixed, every step after the second was refused and waited 20-40 seconds
    # for the window to slide. Halving it roughly halves the wall clock, and
    # the trimmer drops the oldest and largest results first, which are the
    # ones already summarised into the answer.
    context_budget_chars: int = field(
        default_factory=lambda: _int("CONTEXT_BUDGET_CHARS", 4_000))
    tool_result_max_chars: int = field(
        default_factory=lambda: _int("TOOL_RESULT_MAX_CHARS", 6_000))

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.data_dir / "archives"):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key.strip())


settings = Settings()
