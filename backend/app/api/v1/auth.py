"""Authentication endpoints: register, login, refresh, me, logout, password reset."""
from __future__ import annotations

from app.core.apikit import APIRouter, Ctx
from app.schemas.auth import (ForgotPasswordIn, LoginIn, LogoutIn, RefreshIn, RegisterIn, ResetPasswordIn, TokenOut,
                              UserOut)
from app.schemas.common import Message

router = APIRouter("/api/v1/auth", tags=["auth"])


@router.post("/register", auth=False, body=RegisterIn, response=UserOut, status=201, rate="auth")
def register(ctx: Ctx):
    """Create a researcher account (disabled when ALLOW_REGISTRATION=false)."""
    b = ctx.body
    return ctx.state.auth_service.register(b.email, b.password, b.full_name, ctx.client_ip)


@router.post("/login", auth=False, body=LoginIn, response=TokenOut, rate="auth")
def login(ctx: Ctx):
    """Exchange e-mail and password for an access token (15 min) and a rotating refresh token."""
    return ctx.state.auth_service.login(ctx.body.email, ctx.body.password, ctx.client_ip)


@router.post("/refresh", auth=False, body=RefreshIn, response=TokenOut, rate="refresh")
def refresh(ctx: Ctx):
    """Rotate the refresh token and issue a new access token. Re-using an old refresh token revokes the session."""
    return ctx.state.auth_service.refresh(ctx.body.refresh_token, ctx.client_ip)


@router.get("/me", response=UserOut)
def me(ctx: Ctx):
    """The signed-in user."""
    return ctx.user


@router.post("/logout", body=LogoutIn, response=Message)
def logout(ctx: Ctx):
    """Revoke the current access token and (if given) the refresh-token family."""
    ctx.state.auth_service.logout(ctx.user, ctx.body.refresh_token, ctx.client_ip)
    return {"message": "Signed out"}


@router.post("/forgot-password", auth=False, body=ForgotPasswordIn, response=Message, rate="auth")
def forgot_password(ctx: Ctx):
    """E-mail a one-hour reset link (503 if e-mail delivery is not configured). Same answer for unknown addresses."""
    ctx.state.auth_service.forgot_password(ctx.body.email, ctx.client_ip)
    return {"message": "If an account exists for this address, a reset link has been sent"}


@router.post("/reset-password", auth=False, body=ResetPasswordIn, response=Message, rate="auth")
def reset_password(ctx: Ctx):
    """Set a new password with a reset token; all existing sessions are revoked."""
    ctx.state.auth_service.reset_password(ctx.body.token, ctx.body.new_password, ctx.client_ip)
    return {"message": "Password updated; sign in with the new password"}
