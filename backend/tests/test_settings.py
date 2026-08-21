"""
Tests for the per-device settings store.

These exist because saving a setting deadlocked: `save()` acquired a
non-reentrant lock and then called `load()`, which tried to acquire the same
lock. Reads worked fine, so the bug was invisible until someone tried to save
an API key -- from either the CLI or the UI -- and the process hung forever.

Every test here runs under a timeout, because the failure mode is a hang rather
than an exception and a plain assertion would never be reached.
"""

import json
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402


def _fresh_store(tmp: Path):
    """Reload the module against a temporary data directory."""
    settings.data_dir = tmp
    import importlib

    from app import settings_store

    importlib.reload(settings_store)
    return settings_store


def _with_timeout(fn, seconds: float = 10.0):
    """Run fn in a thread and fail if it does not finish. Catches deadlocks."""
    result: dict = {}

    def runner():
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(seconds)

    assert not thread.is_alive(), (
        f"{getattr(fn, '__name__', 'call')} did not finish within {seconds}s "
        "-- almost certainly a lock deadlock."
    )
    if "error" in result:
        raise result["error"]
    return result.get("value")


def test_save_does_not_deadlock():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh_store(Path(tmp))
        _with_timeout(lambda: store.save({"gemini_api_key": "test-key-1234"}))
        assert store.public_view()["api_key_set"] is True


def test_repeated_saves_do_not_deadlock():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh_store(Path(tmp))
        for i in range(5):
            _with_timeout(lambda i=i: store.save({"media_concurrency": i + 1}))
        assert store.public_view()["media_concurrency"] == 5


def test_concurrent_saves_do_not_deadlock():
    """The API server saves from request threads; contention must be safe."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh_store(Path(tmp))

        def hammer():
            for i in range(10):
                store.save({"max_requests_per_minute": i + 1})
                store.load()
                store.public_view()

        threads = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
            assert not t.is_alive(), "concurrent saves deadlocked"


def test_key_is_written_to_disk():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh_store(Path(tmp))
        store.save({"gemini_api_key": "abcd1234"})
        stored = json.loads((Path(tmp) / "config.json").read_text(encoding="utf-8"))
        assert stored["gemini_api_key"] == "abcd1234"


def test_external_change_is_picked_up():
    """
    A key set by the CLI must reach an already-running server.

    They are separate processes, so an in-memory cache that never re-checks the
    file would leave the server believing no key is configured until restart.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh_store(Path(tmp))
        store.save({"gemini_api_key": "first-key"})
        assert store.load()["gemini_api_key"] == "first-key"

        # Simulate the CLI writing the file behind the server's back.
        path = Path(tmp) / "config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["gemini_api_key"] = "second-key"
        path.write_text(json.dumps(payload), encoding="utf-8")
        import os
        import time

        # Ensure the mtime actually differs on coarse-resolution filesystems.
        future = time.time() + 2
        os.utime(path, (future, future))

        assert store.load()["gemini_api_key"] == "second-key"


def test_empty_value_unsets_rather_than_storing_blank():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh_store(Path(tmp))
        store.save({"gemini_api_key": "something"})
        store.save({"gemini_api_key": ""})
        assert store.public_view()["api_key_set"] is False


def test_unknown_keys_are_rejected():
    """The browser must not be able to set arbitrary server options."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh_store(Path(tmp))
        try:
            store.save({"data_dir": "/etc"})
        except ValueError as exc:
            assert "data_dir" in str(exc)
        else:
            raise AssertionError("expected unknown key to be rejected")


def test_key_is_never_exposed_in_public_view():
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh_store(Path(tmp))
        secret = "AQ.SuperSecretValue9999"
        store.save({"gemini_api_key": secret})
        view = store.public_view()
        assert secret not in json.dumps(view)
        assert view["api_key_hint"] == "...9999"


def test_external_change_reaches_the_live_settings_object():
    """
    An externally written key must reach Settings, not just the cache.

    This is the exact split that shipped broken: /api/settings reported a key
    was configured (it reads the file) while the Gemini client saw none (it
    reads Settings), so the app claimed to be ready and then refused to run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = _fresh_store(Path(tmp))
        from app.config import settings as live

        store.save({"gemini_api_key": "key-one"})
        assert live.api_key == "key-one"

        path = Path(tmp) / "config.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["gemini_api_key"] = "key-two"
        path.write_text(json.dumps(payload), encoding="utf-8")

        import os
        import time

        future = time.time() + 2
        os.utime(path, (future, future))

        store.load()                      # any read path should heal it
        assert live.api_key == "key-two"
        assert live.has_api_key is True
        assert store.public_view()["api_key_set"] is True



if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
