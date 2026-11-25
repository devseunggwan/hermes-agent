"""A local reply's files ride the durable transcript, not a second send.

Viewers render ``MEDIA:`` tags and ``::preview{file=…}`` paths from the transcript, so the reply
pipeline re-sending them through ``send_document`` only hit the base fallback and logged a false
"native file send unavailable". A direct ``send_document`` (``/save``) is not in any transcript, so
its warning is a true non-delivery and must stay.
"""
import logging
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.session_local import LocalSessionAdapter

FALLBACK = "send unavailable"


def _adapter():
    return LocalSessionAdapter(SimpleNamespace(runner=None, sessions={}))


def _event():
    source = SessionSource(platform=Platform.LOCAL, chat_id="local-1", chat_type="dm")
    return MessageEvent(text="hi", message_type=MessageType.TEXT, source=source, message_id="1")


def _files(tmp_path, monkeypatch):
    doc, page, png = tmp_path / "brief.md", tmp_path / "draft.html", tmp_path / "chart.png"
    doc.write_text("# brief\n")
    page.write_text("<p>draft</p>")
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    monkeypatch.setattr(BasePlatformAdapter, "filter_media_delivery_paths",
                        staticmethod(lambda files, **_k: files))
    return doc, page, png


@pytest.mark.asyncio
async def test_reply_files_are_acknowledged_without_a_fallback_send(tmp_path, monkeypatch, caplog):
    doc, page, png = _files(tmp_path, monkeypatch)
    adapter = _adapter()
    reply = f"Wrote {doc}\n\n::preview{{file=\"{page}\"}}\n\nMEDIA:{png}"
    extracted = await adapter._extract_response_content(reply, _event(), "k", is_ephemeral_response=False)
    assert extracted.local_files or extracted.media_files  # positive control: the pipeline found files

    outcomes = []
    with caplog.at_level(logging.WARNING):
        await adapter._deliver_attachments(_event(), extracted, {}, anything_sent=False,
                                           record_delivery=outcomes.append)

    assert FALLBACK not in caplog.text
    assert outcomes and all(r.success for r in outcomes)
    assert len(outcomes) == len(extracted.local_files) + len(extracted.media_files)


@pytest.mark.asyncio
async def test_direct_send_document_still_logs_the_non_delivery(tmp_path, monkeypatch, caplog):
    doc, _page, _png = _files(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING):
        await _adapter().send_document(chat_id="local-1", file_path=str(doc), file_name="brief.md")
    assert FALLBACK in caplog.text
