"""Identity helpers for the Streamlit app.

Databricks Apps inject the OAuth-authenticated user's identity into every
inbound request as ``X-Forwarded-*`` headers. Streamlit exposes those via
``st.context.headers`` (>=1.37). The notebook-side ``get_current_username``
in ``customer_retention.analysis.auto_explorer.session`` is NOT reusable
here -- it reads ``DATABRICKS_USERNAME`` (set only inside notebook tasks)
and falls back to ``getpass.getuser`` (the App service principal, identical
for everyone). The App needs request-scoped resolution.

Resolution order for ``current_user_email``:
  1. ``X-Forwarded-Email`` request header  (production)
  2. ``st.experimental_user.email``         (Streamlit-native fallback)
  3. ``CR_USER`` env var                    (local dev / tests)
  4. ``None``                               (caller decides whether to block)

``current_user_handle`` returns the local-part of the email (``jane.doe``
from ``jane.doe@churnkit.com``) for compact display in tables and button
labels. Falls back to ``getpass.getuser()`` when no email is resolvable so
local dev still works without setting any env vars.

Streamlit is imported lazily so this module loads cleanly in the test
venv (which intentionally omits the live-app dependencies).
"""
from __future__ import annotations

import getpass
import os


def _from_request_header() -> str | None:
    try:
        import streamlit as st  # noqa: PLC0415 - lazy so the module imports without streamlit
        email = st.context.headers.get("X-Forwarded-Email")
    except Exception:
        return None
    return email.strip().lower() if email else None


def _from_streamlit_user() -> str | None:
    try:
        import streamlit as st  # noqa: PLC0415
        email = getattr(st.experimental_user, "email", None)
    except Exception:
        return None
    return email.strip().lower() if email else None


def _from_env() -> str | None:
    raw = os.environ.get("CR_USER")
    return raw.strip().lower() if raw else None


def current_user_email() -> str | None:
    """Return the logged-in user's email, or ``None`` when not resolvable."""
    return _from_request_header() or _from_streamlit_user() or _from_env()


def current_user_handle() -> str:
    """Short display handle for the logged-in user.

    ``jane.doe@churnkit.com`` -> ``jane.doe``. Falls back to the OS username
    when no email is available so local dev without ``CR_USER`` still
    renders something deterministic.
    """
    email = current_user_email()
    if email and "@" in email:
        return email.split("@", 1)[0]
    if email:
        return email
    return getpass.getuser()
