"""Shared response models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class Message(BaseModel):
    message: str


class PageMeta(BaseModel):
    page: int
    page_size: int
    total: int
    pages: int


class PageQuery(BaseModel):
    page: int = Field(1, ge=1, le=100000)
    page_size: int = Field(20, ge=1, le=100)


def page_meta(page: int, page_size: int, total: int) -> PageMeta:
    return PageMeta(page=page, page_size=page_size, total=total, pages=max(1, -(-total // page_size)))
