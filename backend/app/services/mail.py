"""Outgoing e-mail (password reset). Backends: disabled (feature unavailable), smtp, file (development: writes .eml
files to MAIL_FILE_DIR instead of sending)."""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage

from app.core.security import new_id

log = logging.getLogger("lti.mail")


def available(settings) -> bool:
    if settings.mail_backend == "smtp":
        return bool(settings.smtp_host)
    return settings.mail_backend == "file"


def send(settings, to: str, subject: str, body: str) -> bool:
    if not available(settings):
        return False
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = settings.mail_from, to, subject
    msg.set_content(body)
    if settings.mail_backend == "file":
        settings.mail_file_dir.mkdir(parents=True, exist_ok=True)
        (settings.mail_file_dir / f"{new_id('mail_')}.eml").write_bytes(bytes(msg))
        return True
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as s:
            if settings.smtp_starttls:
                s.starttls(context=ssl.create_default_context())
            if settings.smtp_user:
                s.login(settings.smtp_user, settings.smtp_password)
            s.send_message(msg)
        return True
    except (OSError, smtplib.SMTPException):
        log.exception("e-mail delivery failed")
        return False
