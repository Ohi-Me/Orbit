"""
Test environment setup.

THIS FILE MUST RUN BEFORE ANY TEST MODULE IS IMPORTED, and pytest guarantees
that: conftest.py is loaded during collection, ahead of the test modules
themselves. That ordering is the entire point.

The bug it fixes: app.core.db builds its SQLAlchemy engine at IMPORT time from
get_settings().database_url. Setting DATABASE_URL at the top of an individual
test module is therefore too late whenever some earlier-collected module has
already imported app.core.db -- the engine is already bound, and the suite
silently reads and writes the real development database instead of a throwaway
one. The symptom is a test that passes alone and fails in the full suite (a
signup returning 409 because a user from a previous run is still there), which
is a miserable thing to debug and an easy thing to prevent.

Every value below is forced rather than defaulted, so a variable left over in
the developer's shell cannot change what the suite does.
"""

from __future__ import annotations

import os
import tempfile

# --- storage ---------------------------------------------------------------
# A fresh directory per invocation, so runs never inherit each other's state.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="orbit-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB_DIR}/test.db"

# --- hermeticity -----------------------------------------------------------
# No model downloads, no market-data calls, no tracking server. The suite must
# pass on a machine with no network and no credentials, and must never depend
# on a live third-party service being up.
os.environ["ALLOW_MODEL_DOWNLOAD"] = "0"
os.environ["ALLOW_LIVE_MARKET_DATA"] = "0"
os.environ["ALLOW_LIVE_FILINGS"] = "0"
os.environ["ALLOW_LIVE_MACRO"] = "0"
os.environ["ENABLE_MLFLOW"] = "0"

# --- auth ------------------------------------------------------------------
# A fixed non-default secret keeps token tests deterministic and stops the
# "using the development default" warning from firing during the suite.
os.environ["JWT_SECRET"] = "test-only-secret-not-used-anywhere-else"

# --- CORS ------------------------------------------------------------------
# Cleared so the CORS tests start from a known state; they set it themselves.
os.environ.pop("CORS_ORIGIN_REGEX", None)


def pytest_report_header(config):
    return f"orbit: isolated test database at {_TEST_DB_DIR}"
