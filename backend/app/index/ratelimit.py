"""
Client-side rate limiting, so a large ingest can run on the free tier.

Google no longer publishes fixed free-tier numbers -- limits vary by account
and are shown per-project in AI Studio -- so nothing here assumes a specific
quota. Instead:

  * requests are spaced to a configurable requests-per-minute ceiling
  * a daily counter is persisted, so a cap survives restarts
  * repeated throttling backs the rate off on its own

The daily cap is what makes free-tier use practical rather than frustrating.
When it is reached the run stops cleanly instead of grinding through hundreds
of failing retries, and because every described file and embedded chunk is
cached by content hash, running again tomorrow resumes exactly where it left
off and re-pays for nothing.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass
from datetime import date

from ..config import settings


def _is_daily_quota(message: str) -> bool:
    """
    Distinguish "too fast" from "done for today".

    A per-minute refusal must never be read as a daily one: adopting its limit
    as the day's budget ends the run on the first 429 that a short wait would
    have cleared. The embedding quota is exactly that trap --
    `embed_content_free_tier_requests` with quotaId
    `EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier` is a
    *minute* limit whose metric name still contains "free_tier_requests", so
    per-minute is checked first and wins.
    """
    lower = message.lower()
    squashed = lower.replace(" ", "").replace("_", "")
    if "perminute" in squashed:
        return False
    return ("perday" in squashed
            or "requests_per_day" in lower
            or "free_tier_requests" in lower)


def retry_after(message: str) -> float | None:
    """
    How long the API asked us to wait, in seconds.

    A per-minute 429 states its own wait ("Please retry in 15.97s",
    "retryDelay: '15s'"). Exponential backoff guesses at that number and
    usually guesses low, which spends attempts rediscovering the same wall.
    """
    for pattern in (r"retry in\s*([\d.]+)s",
                    r"retrydelay['\"]?:\s*['\"]?([\d.]+)s"):
        found = re.search(pattern, message, re.IGNORECASE)
        if found:
            try:
                return float(found.group(1))
            except ValueError:
                continue
    return None


def _extract_limit(message: str) -> int | None:
    """
    Pull the real daily limit out of Google's 429 body.

    It reports e.g. "limit: 20" and "quotaValue: '20'". Believing the API over
    a configured guess is the difference between one wasted request and a
    hundred.
    """
    for pattern in (r"limit:\s*(\d+)", r"quotaValue['\"]?:\s*['\"]?(\d+)"):
        found = re.search(pattern, message)
        if found:
            try:
                return int(found.group(1))
            except ValueError:
                continue
    return None


def credential_id(key: str | None) -> str:
    """
    A stable, non-reversible label for one API key.

    Usage is metered by the provider per project, so it must be counted per
    credential here too. Without this, pasting a fresh key on the Settings tab
    changed nothing: the limiter went on refusing requests because *the day*
    was spent, when what was actually spent was the previous key's quota.

    Hashed rather than stored, and truncated -- this ends up in a local table
    and in log lines, and neither is a place for an API key.
    """
    if not key:
        return "nokey"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class DailyQuotaReached(RuntimeError):
    """Raised when the configured daily request cap is hit."""


@dataclass
class UsageSnapshot:
    requests_today: int
    daily_cap: int | None
    rpm_cap: int | None
    effective_rpm: float
    throttle_events: int
    # Which model this describes. None means "every model summed", which is a
    # headline figure only -- never the number to decide a specific job on.
    model: str | None = None
    per_model: dict[str, dict] | None = None

    def as_dict(self) -> dict:
        return {
            "requests_today": self.requests_today,
            "daily_cap": self.daily_cap,
            "rpm_cap": self.rpm_cap,
            "effective_rpm": round(self.effective_rpm, 2),
            "throttle_events": self.throttle_events,
            "remaining_today": (
                max(0, self.daily_cap - self.requests_today)
                if self.daily_cap else None
            ),
            "model": self.model,
            "per_model": self.per_model or {},
        }


class _ModelBudget:
    """Per-model state: what has been spent, how fast, and any stated ceiling."""

    __slots__ = ("count", "last_request", "penalty", "stated_limit", "loaded")

    def __init__(self) -> None:
        self.count = 0
        self.last_request = 0.0
        self.penalty = 0.0                  # extra spacing added after a 429
        self.stated_limit: int | None = None  # read out of a daily 429
        self.loaded = False


class RateLimiter:
    """
    Spaces outbound requests and tracks daily usage, **per model**.

    Per model, because a single shared budget conflates limits that differ by
    two orders of magnitude, and the failure is not theoretical: a 429 from the
    vision model (genuinely ~20/day) set the global cap to 20, which then
    refused the embedding model (~1,000/day) for the rest of the day. Semantic
    search went dark, the agent fell back to keyword-only, found nothing for
    paraphrased questions, and re-searched until it hit the step limit -- eight
    Groq calls, each waiting out a rate limit, for one question that should have
    taken two. A quota that one model spends must not be a quota another model
    cannot use.

    It also keeps Groq speech transcription off the Gemini budget entirely,
    which it had been consuming for no reason at all.

    Deliberately conservative within a model: it is far better to take longer
    than to burn a day's quota on retries that were always going to fail.
    """

    DEFAULT = "default"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day = date.today()
        self._budgets: dict[tuple[str, str], _ModelBudget] = {}
        self._throttle_events = 0
        # Tests drive the counters directly and must not write to the device's
        # real usage table; nothing else turns this off.
        self._persist_enabled = True

    # --- state -------------------------------------------------------------
    def _budget(self, model: str, credential: str | None = None) -> _ModelBudget:
        """
        This model-and-credential's budget, read from disk when first touched.

        Keyed by both because a quota belongs to a key, not to a calendar day:
        swapping in another key must give you that key's allowance, and must not
        hand you the spent one back.
        """
        cred = credential if credential is not None else self.current_credential()
        slot = (model, cred)
        budget = self._budgets.get(slot)
        if budget is None:
            budget = _ModelBudget()
            self._budgets[slot] = budget
        if not budget.loaded:
            budget.loaded = True
            self._load(model, cred, budget)
        return budget

    @staticmethod
    def current_credential() -> str:
        """Fingerprint of the key Gemini calls are currently using."""
        return credential_id(settings.api_key)

    def _load(self, model: str, credential: str, budget: _ModelBudget) -> None:
        try:
            from ..db import get_cache_connection

            conn = get_cache_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage_by_key (
                    day        DATE,
                    model      VARCHAR,
                    credential VARCHAR,
                    requests   BIGINT,
                    PRIMARY KEY (day, model, credential)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_limit_by_key (
                    day         DATE,
                    model       VARCHAR,
                    credential  VARCHAR,
                    limit_value BIGINT,
                    PRIMARY KEY (day, model, credential)
                )
            """)
            row = conn.execute(
                "SELECT requests FROM api_usage_by_key "
                "WHERE day = ? AND model = ? AND credential = ?",
                [self._day, model, credential],
            ).fetchone()
            budget.count = int(row[0]) if row else 0

            # A daily limit this model stated earlier today still applies today.
            # Without it a restart forgets, the UI reports a budget the provider
            # has already refused, and the next run spends a request to
            # rediscover it.
            stated = conn.execute(
                "SELECT limit_value FROM api_limit_by_key "
                "WHERE day = ? AND model = ? AND credential = ?",
                [self._day, model, credential],
            ).fetchone()
            if stated and stated[0]:
                budget.stated_limit = int(stated[0])
        except Exception:  # noqa: BLE001 - usage tracking must never block work
            budget.count = 0

    def _persist(self, model: str, credential: str,
                 budget: _ModelBudget) -> None:
        """
        Write the model's count out on *every* request.

        This used to fire only on multiples of 20, under a cap of 20 -- so the
        row was first written by the request that had already spent the budget,
        and every earlier one was invisible to the next process. The write is a
        keyed upsert into a local file, a few times a minute at most given the
        rpm ceiling above it: cheap enough to do at the only frequency that is
        actually correct.
        """
        if not self._persist_enabled:
            return
        try:
            from ..db import get_cache_connection

            get_cache_connection().execute(
                "INSERT INTO api_usage_by_key VALUES (?, ?, ?, ?) "
                "ON CONFLICT (day, model, credential) DO UPDATE SET "
                "requests = excluded.requests",
                [self._day, model, credential, budget.count],
            )
        except Exception:  # noqa: BLE001
            pass

    def _persist_stated_limit(self, model: str, credential: str,
                              value: int) -> None:
        """
        Remember, for today only, a daily limit this model stated.

        Day-scoped on purpose: it must outlive a restart, because the provider's
        refusal does, but it must not outlive the day -- and it is never written
        into the user's config, where a transient 429 would quietly become a
        permanent setting.
        """
        if not self._persist_enabled:
            return
        try:
            from ..db import get_cache_connection

            conn = get_cache_connection()
            # A 429 can be the first thing that touches this table.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_limit_by_key (
                    day         DATE,
                    model       VARCHAR,
                    credential  VARCHAR,
                    limit_value BIGINT,
                    PRIMARY KEY (day, model, credential)
                )
            """)
            conn.execute(
                "INSERT INTO api_limit_by_key VALUES (?, ?, ?, ?) "
                "ON CONFLICT (day, model, credential) DO UPDATE SET limit_value = "
                "least(excluded.limit_value, api_limit_by_key.limit_value)",
                [self._day, model, credential, value],
            )
        except Exception:  # noqa: BLE001
            pass

    def _roll_day(self) -> None:
        today = date.today()
        if today != self._day:
            self._day = today
            self._budgets.clear()

    def _cap_for(self, budget: _ModelBudget) -> int:
        """
        The effective daily cap: the lower of what the API said and what is
        configured.

        Adopted downwards only. Raising a configured cap because a 429 quoted a
        bigger number would overrule a deliberate setting, and only the lower
        direction is something the API actually told us.
        """
        configured = settings.max_requests_per_day or 0
        stated = budget.stated_limit or 0
        if configured and stated:
            return min(configured, stated)
        return stated or configured

    # --- public API ----------------------------------------------------
    def acquire(self, model: str | None = None,
                credential: str | None = None) -> None:
        """Block until another request is allowed. Raises when the day is up."""
        key = model or self.DEFAULT
        with self._lock:
            self._roll_day()
            cred = (credential if credential is not None
                    else self.current_credential())
            budget = self._budget(key, cred)

            cap = self._cap_for(budget)
            if cap and budget.count >= cap:
                raise DailyQuotaReached(
                    f"Reached the daily limit of {cap} API requests for "
                    f"{key} on this API key. Everything processed so far is "
                    "cached, so running this again tomorrow continues from "
                    "here without repeating work. You can also raise "
                    "MAX_REQUESTS_PER_DAY, or switch to a different API key on "
                    "the Settings tab -- each key has its own quota."
                )

            rpm = settings.max_requests_per_minute
            if rpm and rpm > 0:
                min_gap = 60.0 / rpm + budget.penalty
                wait = min_gap - (time.monotonic() - budget.last_request)
                if wait > 0:
                    time.sleep(wait)

            budget.last_request = time.monotonic()
            budget.count += 1
            self._persist(key, cred, budget)

    def note_throttled(self, message: str = "", model: str | None = None,
                       credential: str | None = None) -> None:
        """
        Record that the API pushed back on this model.

        Adds permanent extra spacing for the rest of the run. Retrying at the
        same rate after a 429 just produces more 429s.

        A *daily* quota refusal is different in kind from a per-minute one:
        waiting will not help until tomorrow. Google states the real number in
        the error, so we adopt it rather than continuing to guess -- otherwise
        every later request spends retries rediscovering the same wall.
        """
        key = model or self.DEFAULT
        with self._lock:
            self._throttle_events += 1
            cred = (credential if credential is not None
                    else self.current_credential())
            budget = self._budget(key, cred)
            budget.penalty = min(budget.penalty + 1.0, 15.0)

            if not _is_daily_quota(message):
                return

            actual = _extract_limit(message)
            if actual is not None and actual > 0:
                budget.stated_limit = actual
                # Whatever we thought we had spent, the account says we are out.
                budget.count = max(budget.count, actual)
                self._persist_stated_limit(key, cred, actual)
            else:
                budget.count = max(budget.count, self._cap_for(budget) or 1)
            self._persist(key, cred, budget)

    def snapshot(self, model: str | None = None) -> UsageSnapshot:
        """
        Usage for one model, or every model summed when none is named.

        The summed view is for a headline figure. Anything deciding whether a
        specific job can run should name its model, or it will be told about a
        budget that has nothing to do with it.
        """
        with self._lock:
            self._roll_day()
            rpm = settings.max_requests_per_minute

            if model:
                budget = self._budget(model)
                return UsageSnapshot(
                    requests_today=budget.count,
                    daily_cap=self._cap_for(budget) or None,
                    rpm_cap=rpm or None,
                    effective_rpm=(60.0 / (60.0 / rpm + budget.penalty)
                                   if rpm else 0.0),
                    throttle_events=self._throttle_events,
                    model=model,
                )

            for known in (settings.embed_model, settings.vision_model):
                self._budget(known)     # so a fresh process reports real totals
            spent = sum(b.count for b in self._budgets.values())
            worst = max((b.penalty for b in self._budgets.values()), default=0.0)
            return UsageSnapshot(
                requests_today=spent,
                daily_cap=settings.max_requests_per_day or None,
                rpm_cap=rpm or None,
                effective_rpm=60.0 / (60.0 / rpm + worst) if rpm else 0.0,
                throttle_events=self._throttle_events,
                model=None,
                per_model={
                    name: {
                        "requests_today": b.count,
                        "daily_cap": self._cap_for(b) or None,
                        "remaining_today": (
                            max(0, self._cap_for(b) - b.count)
                            if self._cap_for(b) else None
                        ),
                    }
                    for (name, _cred), b in sorted(self._budgets.items())
                },
            )

    def flush(self) -> None:
        with self._lock:
            for (name, cred), budget in self._budgets.items():
                self._persist(name, cred, budget)


limiter = RateLimiter()
