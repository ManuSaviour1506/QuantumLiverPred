"""User profile and administration endpoints."""
from __future__ import annotations

from app.core.apikit import APIRouter, Ctx
from app.schemas.auth import AdminUserUpdateIn, ChangePasswordIn, UpdateMeIn, UserListQuery, UserOut, UserPage
from app.schemas.common import Message, page_meta

router = APIRouter("/api/v1/users", tags=["users"])


@router.get("/me", response=UserOut)
def get_me(ctx: Ctx):
    """Profile of the signed-in user."""
    return ctx.user


@router.patch("/me", body=UpdateMeIn, response=UserOut)
def update_me(ctx: Ctx):
    """Update display name and preferences (theme, default model, page size)."""
    return ctx.state.auth_service.update_me(ctx.user, ctx.body.full_name, ctx.body.preferences)


@router.post("/me/password", body=ChangePasswordIn, response=Message, rate="auth")
def change_password(ctx: Ctx):
    """Change password (requires the current password); other sessions are signed out."""
    ctx.state.auth_service.change_password(ctx.user, ctx.body.current_password, ctx.body.new_password, ctx.client_ip)
    return {"message": "Password changed"}


@router.get("", auth=("admin",), query=UserListQuery, response=UserPage)
def list_users(ctx: Ctx):
    """All users (administrators only)."""
    q = ctx.query
    items, total = ctx.state.auth_service.list_users(q.q, q.page, q.page_size)
    return {"items": items, "meta": page_meta(q.page, q.page_size, total)}


@router.patch("/{user_id}", auth=("admin",), body=AdminUserUpdateIn, response=UserOut)
def admin_update_user(ctx: Ctx):
    """Change a user's role or active flag (administrators only)."""
    return ctx.state.auth_service.admin_update(ctx.user, ctx.path["user_id"], ctx.body.role, ctx.body.is_active,
                                               ctx.client_ip)
