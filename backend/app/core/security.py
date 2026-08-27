"""
Authentication and authorization.

DESIGN NOTES
  * Passwords are hashed with bcrypt via passlib. Never stored, never logged.
  * Tokens are short-lived JWTs. There is no refresh-token rotation here and
    that is a stated limitation rather than an oversight -- adding one without
    a revocation store would be security theatre.
  * AUTH_REQUIRED defaults to false so the platform runs locally with zero
    setup, but `effective_capabilities()` reports whether the JWT secret is
    still the development default, and /api/health surfaces it. A deployment
    running on the default secret is visibly misconfigured rather than quietly
    insecure.
  * Run ownership is enforced at the query layer: a user can only read runs
    whose user_id matches their own, checked in the route rather than trusted
    from a client-supplied id.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_db
from app.core.models import User

bearer_scheme = HTTPBearer(auto_error=False)

# bcrypt is used directly rather than through passlib. passlib 1.7.x probes
# `bcrypt.__about__.__version__`, which bcrypt 4+ removed, so its backend
# detection fails and it mis-handles the 72-byte limit. The direct API is
# smaller, maintained, and does exactly what is needed here.
_BCRYPT_ROUNDS = 12


def hash_password(password: str) -> str:
    """Hash with bcrypt.

    bcrypt silently truncates input beyond 72 bytes, which would let two
    different long passwords authenticate each other. Rejecting is safer than
    truncating, so the limit is enforced rather than absorbed.
    """
    raw = password.encode("utf-8")
    if len(raw) > 72:
        raise ValueError("Password must be at most 72 bytes.")
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=_BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        raw = plain.encode("utf-8")
        if len(raw) > 72:
            return False
        return bcrypt.checkpw(raw, hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: str, email: str) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {type(e).__name__}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    """Resolve the caller. Returns None for anonymous when AUTH_REQUIRED is off."""
    settings = get_settings()

    if creds is None or not creds.credentials:
        if settings.auth_required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None

    payload = decode_token(creds.credentials)
    user = db.execute(select(User).where(User.id == payload.get("sub"))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive.")
    return user


def require_user(user: User | None = Depends(get_current_user)) -> User:
    """For routes that must have a named human -- approvals, above all.

    An approval without an identity is not an approval; it is an unattributed
    state change, which defeats the point of a human gate.
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="This action requires an authenticated user. Approvals must be "
            "attributable to a named person.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def owns_or_404(run, user: User | None):
    """Ownership check used by every run-scoped route.

    Returns 404 rather than 403 for a run owned by someone else, so the API
    does not confirm the existence of resources the caller cannot see.
    """
    from fastapi import HTTPException as _HTTPException

    if run is None:
        raise _HTTPException(status_code=404, detail="Run not found.")
    if run.user_id is not None and (user is None or run.user_id != user.id):
        raise _HTTPException(status_code=404, detail="Run not found.")
    return run
