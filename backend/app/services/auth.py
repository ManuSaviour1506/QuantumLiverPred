"""Users, roles and authentication: registration, login with lock-out, refresh-token rotation with reuse detection,
logout (refresh revocation + access-token deny list), password change and reset."""
from __future__ import annotations

import datetime as dt

from app.core.errors import ApiError, conflict, forbidden, not_found, unauthorized
from app.core.security import (PasswordHasher, create_access_token, decode_access_token, dummy_verify, iso,
                               new_id, new_opaque_token, parse_iso, password_problems, token_hash, utcnow)
from app.db.database import Connection, loads
from app.services import audit, mail

ROLE_IDS = {"admin": "role_admin", "researcher": "role_researcher", "viewer": "role_viewer"}

USER_SELECT = ("SELECT u.id, u.email, u.full_name, u.password_hash, u.is_active, u.failed_logins, u.locked_until, "
               "u.last_login_at, u.preferences, u.created_at, u.updated_at, r.name AS role "
               "FROM users u JOIN roles r ON r.id = u.role_id")


def public_user(row: dict) -> dict:
    return {"id": row["id"], "email": row["email"], "full_name": row["full_name"], "role": row["role"],
            "is_active": bool(row["is_active"]), "created_at": row["created_at"],
            "last_login_at": row.get("last_login_at"), "preferences": loads(row.get("preferences"), {})}


class AuthService:
    def __init__(self, settings, db):
        self.settings, self.db = settings, db
        self.hasher = PasswordHasher(settings.argon2_memory_kib, settings.argon2_iterations)

    # ------------------------------------------------------------------ tokens
    def decode(self, token: str) -> dict:
        s = self.settings
        return decode_access_token(token, s.jwt_secret, s.jwt_issuer, s.jwt_algorithm)

    def user_from_claims(self, claims: dict) -> dict:
        with self.db.session() as c:
            if c.scalar("SELECT 1 FROM token_denylist WHERE jti = ?", (claims["jti"],)):
                raise unauthorized("Session has been logged out")
            row = c.one(USER_SELECT + " WHERE u.id = ?", (claims["sub"],))
        if row is None or not row["is_active"]:
            raise unauthorized("Account not found or disabled")
        user = public_user(row)
        user["_jti"], user["_exp"] = claims["jti"], claims["exp"]
        return user

    def _issue(self, c: Connection, row: dict, family_id: str | None = None) -> dict:
        s = self.settings
        access, exp, _jti = create_access_token(row["id"], row["role"], s.jwt_secret, s.access_token_minutes,
                                                s.jwt_issuer, s.jwt_algorithm)
        refresh, rhash = new_opaque_token()
        rid = new_id("rt_")
        now = utcnow()
        r_exp = now + dt.timedelta(days=s.refresh_token_days)
        c.run("INSERT INTO refresh_tokens (id, user_id, token_hash, family_id, created_at, expires_at) "
              "VALUES (?, ?, ?, ?, ?, ?)", (rid, row["id"], rhash, family_id or new_id("rf_"), iso(now), iso(r_exp)))
        return {"access_token": access, "token_type": "bearer", "expires_in": s.access_token_minutes * 60,
                "refresh_token": refresh, "refresh_expires_at": iso(r_exp), "user": public_user(row), "_rid": rid}

    # ------------------------------------------------------------------ users
    def create_user(self, c: Connection, email: str, password: str, full_name: str, role: str = "researcher") -> dict:
        problems = password_problems(password, self.settings.password_min_length)
        if problems:
            raise ApiError(422, "weak_password", "Password " + "; ".join(problems),
                           [{"field": "password", "message": p} for p in problems])
        if c.scalar("SELECT 1 FROM users WHERE email = ?", (email,)):
            raise conflict("An account with this e-mail address already exists")
        now = iso(utcnow())
        uid = new_id("usr_")
        c.run("INSERT INTO users (id, email, full_name, password_hash, role_id, is_active, created_at, updated_at) "
              "VALUES (?, ?, ?, ?, ?, 1, ?, ?)", (uid, email, full_name, self.hasher.hash(password), ROLE_IDS[role],
                                                  now, now))
        return c.one(USER_SELECT + " WHERE u.id = ?", (uid,))

    def register(self, email: str, password: str, full_name: str, ip: str | None) -> dict:
        if not self.settings.allow_registration:
            raise forbidden("Self-registration is disabled on this server; ask an administrator for an account")
        with self.db.session() as c:
            with c.transaction():
                row = self.create_user(c, email, password, full_name, "researcher")
                audit.log(c, "user.registered", row["id"], "user", row["id"], ip, secret=self.settings.jwt_secret)
        return public_user(row)

    def login(self, email: str, password: str, ip: str | None) -> dict:
        s = self.settings
        with self.db.session() as c:
            row = c.one(USER_SELECT + " WHERE u.email = ?", (email,))
            if row is None:
                dummy_verify(self.hasher)
                raise unauthorized("Incorrect e-mail address or password")
            locked = parse_iso(row["locked_until"])
            if locked and locked > utcnow():
                raise ApiError(423, "account_locked", "Too many failed sign-in attempts; try again later",
                               {"locked_until": row["locked_until"]})
            if not self.hasher.verify(password, row["password_hash"]):
                fails = int(row["failed_logins"]) + 1
                lock = iso(utcnow() + dt.timedelta(minutes=s.lockout_minutes)) if fails >= s.max_failed_logins else None
                c.run("UPDATE users SET failed_logins = ?, locked_until = ? WHERE id = ?",
                      (0 if lock else fails, lock, row["id"]))
                audit.log(c, "auth.login_failed", row["id"], "user", row["id"], ip,
                          {"locked": bool(lock)}, secret=s.jwt_secret)
                c.commit()
                raise unauthorized("Incorrect e-mail address or password")
            if not row["is_active"]:
                raise forbidden("This account is disabled")
            with c.transaction():
                upd = {"h": self.hasher.hash(password)} if self.hasher.needs_rehash(row["password_hash"]) else {}
                c.run("UPDATE users SET failed_logins = 0, locked_until = NULL, last_login_at = ?" +
                      (", password_hash = ?" if upd else "") + " WHERE id = ?",
                      (iso(utcnow()), *([upd["h"]] if upd else []), row["id"]))
                row = c.one(USER_SELECT + " WHERE u.id = ?", (row["id"],))
                out = self._issue(c, row)
                audit.log(c, "auth.login", row["id"], "user", row["id"], ip, secret=s.jwt_secret)
        out.pop("_rid", None)
        return out

    def refresh(self, token: str, ip: str | None) -> dict:
        h = token_hash(token)
        with self.db.session() as c:
            rt = c.one("SELECT id, user_id, family_id, expires_at, revoked_at FROM refresh_tokens WHERE token_hash = ?",
                       (h,))
            if rt is None:
                raise unauthorized("Invalid refresh token")
            if rt["revoked_at"]:
                # A rotated token was used again: possible theft. Revoke the whole family.
                c.run("UPDATE refresh_tokens SET revoked_at = ? WHERE family_id = ? AND revoked_at IS NULL",
                      (iso(utcnow()), rt["family_id"]))
                audit.log(c, "auth.refresh_reuse_detected", rt["user_id"], "user", rt["user_id"], ip,
                          secret=self.settings.jwt_secret)
                c.commit()
                raise unauthorized("Refresh token already used; please sign in again")
            if parse_iso(rt["expires_at"]) <= utcnow():
                raise unauthorized("Refresh token expired")
            row = c.one(USER_SELECT + " WHERE u.id = ?", (rt["user_id"],))
            if row is None or not row["is_active"]:
                raise unauthorized("Account not found or disabled")
            with c.transaction():
                out = self._issue(c, row, rt["family_id"])
                c.run("UPDATE refresh_tokens SET revoked_at = ?, replaced_by = ? WHERE id = ?",
                      (iso(utcnow()), out["_rid"], rt["id"]))
        out.pop("_rid", None)
        return out

    def logout(self, user: dict, refresh_token: str | None, ip: str | None) -> None:
        with self.db.session() as c:
            with c.transaction():
                exp = dt.datetime.fromtimestamp(user["_exp"], dt.timezone.utc)
                c.run("INSERT INTO token_denylist (jti, expires_at) VALUES (?, ?) ON CONFLICT (jti) DO NOTHING",
                      (user["_jti"], iso(exp)))
                if refresh_token:
                    fam = c.scalar("SELECT family_id FROM refresh_tokens WHERE token_hash = ? AND user_id = ?",
                                   (token_hash(refresh_token), user["id"]))
                    if fam:
                        c.run("UPDATE refresh_tokens SET revoked_at = ? WHERE family_id = ? AND revoked_at IS NULL",
                              (iso(utcnow()), fam))
                c.run("DELETE FROM token_denylist WHERE expires_at < ?", (iso(utcnow()),))
                audit.log(c, "auth.logout", user["id"], "user", user["id"], ip, secret=self.settings.jwt_secret)

    def change_password(self, user: dict, current: str, new: str, ip: str | None) -> None:
        with self.db.session() as c:
            row = c.one(USER_SELECT + " WHERE u.id = ?", (user["id"],))
            if row is None or not self.hasher.verify(current, row["password_hash"]):
                raise ApiError(400, "wrong_password", "Current password is incorrect")
            self._set_password(c, row["id"], new)
            audit.log(c, "user.password_changed", row["id"], "user", row["id"], ip, secret=self.settings.jwt_secret)

    def _set_password(self, c: Connection, user_id: str, new: str) -> None:
        problems = password_problems(new, self.settings.password_min_length)
        if problems:
            raise ApiError(422, "weak_password", "Password " + "; ".join(problems),
                           [{"field": "new_password", "message": p} for p in problems])
        with c.transaction():
            c.run("UPDATE users SET password_hash = ?, updated_at = ?, failed_logins = 0, locked_until = NULL "
                  "WHERE id = ?", (self.hasher.hash(new), iso(utcnow()), user_id))
            c.run("UPDATE refresh_tokens SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                  (iso(utcnow()), user_id))

    def forgot_password(self, email: str, ip: str | None) -> bool:
        """Always answers the same way (no account enumeration). Returns False if e-mail is not configured."""
        if not mail.available(self.settings):
            raise ApiError(503, "email_not_configured",
                           "Password reset by e-mail is not configured on this server; contact an administrator")
        with self.db.session() as c:
            row = c.one("SELECT id, email, is_active FROM users WHERE email = ?", (email,))
            if row is None or not row["is_active"]:
                return True
            token, h = new_opaque_token()
            now = utcnow()
            c.run("INSERT INTO password_reset_tokens (id, user_id, token_hash, created_at, expires_at) "
                  "VALUES (?, ?, ?, ?, ?)", (new_id("pr_"), row["id"], h, iso(now), iso(now + dt.timedelta(hours=1))))
            audit.log(c, "auth.password_reset_requested", row["id"], "user", row["id"], ip,
                      secret=self.settings.jwt_secret)
        link = f"{self.settings.frontend_base_url.rstrip('/')}/reset-password?token={token}"
        mail.send(self.settings, row["email"], "Reset your password — Liver Tumour Intelligence Platform",
                  "A password reset was requested for your account.\n\n"
                  f"Open this link within one hour to choose a new password:\n{link}\n\n"
                  "If you did not request this, ignore this message.\n")
        return True

    def reset_password(self, token: str, new: str, ip: str | None) -> None:
        with self.db.session() as c:
            pr = c.one("SELECT id, user_id, expires_at, used_at FROM password_reset_tokens WHERE token_hash = ?",
                       (token_hash(token),))
            if pr is None or pr["used_at"] or parse_iso(pr["expires_at"]) <= utcnow():
                raise ApiError(400, "invalid_reset_token", "This reset link is invalid or has expired")
            self._set_password(c, pr["user_id"], new)
            c.run("UPDATE password_reset_tokens SET used_at = ? WHERE id = ?", (iso(utcnow()), pr["id"]))
            audit.log(c, "auth.password_reset", pr["user_id"], "user", pr["user_id"], ip,
                      secret=self.settings.jwt_secret)

    # ------------------------------------------------------------------ administration
    def list_users(self, q: str | None, page: int, page_size: int) -> tuple[list[dict], int]:
        where, params = "", []
        if q:
            where, params = " WHERE (u.email LIKE ? OR u.full_name LIKE ?)", [f"%{q.lower()}%", f"%{q}%"]
        with self.db.session() as c:
            total = c.scalar("SELECT COUNT(*) FROM users u" + where, params)
            rows = c.all(USER_SELECT + where + " ORDER BY u.created_at DESC LIMIT ? OFFSET ?",
                         (*params, page_size, (page - 1) * page_size))
        return [public_user(r) for r in rows], int(total or 0)

    def admin_update(self, actor: dict, user_id: str, role: str | None, is_active: bool | None,
                     ip: str | None) -> dict:
        with self.db.session() as c:
            row = c.one(USER_SELECT + " WHERE u.id = ?", (user_id,))
            if row is None:
                raise not_found("User")
            if user_id == actor["id"] and (role not in (None, "admin") or is_active is False):
                raise ApiError(400, "self_demotion", "Administrators cannot remove their own admin role or disable "
                                                     "their own account")
            with c.transaction():
                if role is not None:
                    c.run("UPDATE users SET role_id = ?, updated_at = ? WHERE id = ?",
                          (ROLE_IDS[role], iso(utcnow()), user_id))
                if is_active is not None:
                    c.run("UPDATE users SET is_active = ?, updated_at = ? WHERE id = ?",
                          (int(is_active), iso(utcnow()), user_id))
                    if not is_active:
                        c.run("UPDATE refresh_tokens SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
                              (iso(utcnow()), user_id))
                audit.log(c, "user.updated_by_admin", actor["id"], "user", user_id, ip,
                          {"role": role, "is_active": is_active}, secret=self.settings.jwt_secret)
                row = c.one(USER_SELECT + " WHERE u.id = ?", (user_id,))
        return public_user(row)

    def update_me(self, user: dict, full_name: str | None, preferences: dict | None) -> dict:
        with self.db.session() as c:
            if full_name is not None:
                c.run("UPDATE users SET full_name = ?, updated_at = ? WHERE id = ?",
                      (" ".join(full_name.split()), iso(utcnow()), user["id"]))
            if preferences is not None:
                cur = loads(c.scalar("SELECT preferences FROM users WHERE id = ?", (user["id"],)), {})
                cur.update(preferences)
                c.run("UPDATE users SET preferences = ?, updated_at = ? WHERE id = ?", (cur, iso(utcnow()), user["id"]))
            row = c.one(USER_SELECT + " WHERE u.id = ?", (user["id"],))
        return public_user(row)
