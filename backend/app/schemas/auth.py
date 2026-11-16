"""Authentication and user schemas."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import PageMeta, PageQuery

_EMAIL = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
                    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")

Role = Literal["admin", "researcher", "viewer"]


def normalise_email(v: str) -> str:
    v = (v or "").strip().lower()
    if len(v) > 254 or not _EMAIL.match(v):
        raise ValueError("not a valid e-mail address")
    return v


class _EmailModel(BaseModel):
    email: str = Field(..., max_length=254)

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return normalise_email(v)


class RegisterIn(_EmailModel):
    password: str = Field(..., min_length=1, max_length=256)
    full_name: str = Field(..., min_length=1, max_length=120)

    @field_validator("full_name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("full name is required")
        return v


class LoginIn(_EmailModel):
    password: str = Field(..., min_length=1, max_length=256)


class RefreshIn(BaseModel):
    refresh_token: str = Field(..., min_length=20, max_length=200)


class LogoutIn(BaseModel):
    refresh_token: str | None = Field(None, max_length=200)


class ForgotPasswordIn(_EmailModel):
    pass


class ResetPasswordIn(BaseModel):
    token: str = Field(..., min_length=20, max_length=200)
    new_password: str = Field(..., min_length=1, max_length=256)


class ChangePasswordIn(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
    new_password: str = Field(..., min_length=1, max_length=256)


class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    role: Role
    is_active: bool
    created_at: str
    last_login_at: str | None = None
    preferences: dict = Field(default_factory=dict)


class TokenOut(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    refresh_token: str
    refresh_expires_at: str
    user: UserOut


class UpdateMeIn(BaseModel):
    full_name: str | None = Field(None, min_length=1, max_length=120)
    preferences: dict | None = None

    @field_validator("preferences")
    @classmethod
    def _prefs(cls, v):
        if v is None:
            return v
        allowed = {"theme", "default_model", "page_size"}
        bad = set(v) - allowed
        if bad:
            raise ValueError(f"unknown preference(s): {', '.join(sorted(bad))}")
        if "theme" in v and v["theme"] not in ("light", "dark", "system"):
            raise ValueError("theme must be light, dark or system")
        return v


class AdminUserUpdateIn(BaseModel):
    role: Role | None = None
    is_active: bool | None = None


class UserListQuery(PageQuery):
    q: str | None = Field(None, max_length=100)


class UserPage(BaseModel):
    items: list[UserOut]
    meta: PageMeta
