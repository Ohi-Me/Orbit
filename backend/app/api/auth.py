"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import get_db
from app.core.models import User
from app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.schemas import LoginRequest, SignupRequest, TokenResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _user_dict(u: User) -> dict:
    return {"id": u.id, "email": u.email, "display_name": u.display_name}


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    existing = db.execute(select(User).where(User.email == req.email.lower())).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    try:
        hashed = hash_password(req.password)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    user = User(email=req.email.lower(), hashed_password=hashed, display_name=req.display_name)
    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenResponse(
        access_token=create_access_token(user.id, user.email),
        expires_in_minutes=get_settings().jwt_expire_minutes,
        user=_user_dict(user),
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.email == req.email.lower())).scalar_one_or_none()
    # Same message for unknown email and wrong password -- a distinct one would
    # let an attacker enumerate registered accounts.
    if user is None or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="This account is disabled.")

    return TokenResponse(
        access_token=create_access_token(user.id, user.email),
        expires_in_minutes=get_settings().jwt_expire_minutes,
        user=_user_dict(user),
    )


@router.get("/me")
def me(user: User | None = Depends(get_current_user)):
    if user is None:
        return {"authenticated": False, "user": None,
                "note": "Anonymous access is permitted because AUTH_REQUIRED is off."}
    return {"authenticated": True, "user": _user_dict(user)}
