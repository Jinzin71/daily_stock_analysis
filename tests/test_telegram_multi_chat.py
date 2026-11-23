# -*- coding: utf-8 -*-
"""TELEGRAM_CHAT_ID with multiple comma-separated recipients."""
from unittest import mock

from src.notification_sender.telegram_sender import TelegramSender
from tests.test_notification_sender import _config, _response


def _sent_chat_ids(mock_post):
    ids = []
    for call in mock_post.call_args_list:
        payload = call.kwargs.get("json") or call.kwargs.get("data") or {}
        ids.append(str(payload.get("chat_id")))
    return ids


def test_split_chat_ids_variants():
    split = TelegramSender._split_chat_ids
    assert split("5860205304,8564156174") == ["5860205304", "8564156174"]
    assert split(" 5860205304 , 8564156174 ;5860205304\n") == ["5860205304", "8564156174"]
    assert split("-1001234567890") == ["-1001234567890"]
    assert split("") == [] and split(None) == []


@mock.patch("src.notification_sender.telegram_sender.requests.post")
def test_send_to_each_recipient(mock_post):
    mock_post.return_value = _response(200, {"ok": True})
    sender = TelegramSender(_config(telegram_bot_token="BOT", telegram_chat_id="5860205304,8564156174"))
    assert sender.send_to_telegram("hello") is True
    assert _sent_chat_ids(mock_post) == ["5860205304", "8564156174"]


@mock.patch("src.notification_sender.telegram_sender.requests.post")
def test_one_recipient_failure_does_not_block_other(mock_post):
    failed = _response(400)
    failed.text = '{"ok":false,"error_code":400,"description":"Bad Request: chat not found"}'
    mock_post.side_effect = [failed, _response(200, {"ok": True})]
    sender = TelegramSender(_config(telegram_bot_token="BOT", telegram_chat_id="111,222"))
    assert sender.send_to_telegram("hello") is True
    assert _sent_chat_ids(mock_post)[-1] == "222"


@mock.patch("src.notification_sender.telegram_sender.requests.post")
def test_all_recipients_fail_returns_false(mock_post):
    failed = _response(400)
    failed.text = '{"ok":false,"error_code":400,"description":"Bad Request: chat not found"}'
    mock_post.return_value = failed
    sender = TelegramSender(_config(telegram_bot_token="BOT", telegram_chat_id="111,222"))
    assert sender.send_to_telegram("hello") is False


@mock.patch("src.notification_sender.telegram_sender.requests.post")
def test_single_chat_id_unchanged(mock_post):
    mock_post.return_value = _response(200, {"ok": True})
    sender = TelegramSender(_config(telegram_bot_token="BOT", telegram_chat_id="5860205304"))
    assert sender.send_to_telegram("hello") is True
    assert _sent_chat_ids(mock_post) == ["5860205304"]


@mock.patch("src.notification_sender.telegram_sender.requests.post")
def test_photo_sent_to_each_recipient(mock_post):
    mock_post.return_value = _response(200, {"ok": True})
    sender = TelegramSender(_config(telegram_bot_token="BOT", telegram_chat_id="111,222"))
    assert sender._send_telegram_photo(b"png") is True
    assert _sent_chat_ids(mock_post) == ["111", "222"]
