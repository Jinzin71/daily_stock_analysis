# -*- coding: utf-8 -*-
"""Tests for Telegram token normalization and Google News RSS fallback."""
from unittest.mock import MagicMock, patch

from src.config import normalize_telegram_bot_token
from src.search_service import GoogleNewsRssSearchProvider, SearchService

TOKEN = "123456789:AAHabcdefghijklmnopqrstuvwxyz012345"

SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>t</title>
<item>
  <title>Chip stocks rally as Nvidia surges - Reuters</title>
  <link>https://news.google.com/rss/articles/abc</link>
  <pubDate>Wed, 19 Nov 2026 21:05:00 GMT</pubDate>
  <description>&lt;a href="x"&gt;Chip stocks rally as Nvidia surges&lt;/a&gt;&amp;nbsp;&lt;font&gt;Reuters&lt;/font&gt;</description>
  <source url="https://www.reuters.com">Reuters</source>
</item>
<item>
  <title>SOX index slips - CNBC</title>
  <link>https://news.google.com/rss/articles/def</link>
  <pubDate>Wed, 19 Nov 2026 20:00:00 GMT</pubDate>
  <source url="https://www.cnbc.com">CNBC</source>
</item>
</channel></rss>"""


def test_normalize_telegram_bot_token_variants():
    assert normalize_telegram_bot_token(f"  {TOKEN}\n") == TOKEN
    assert normalize_telegram_bot_token(f'"{TOKEN}"') == TOKEN
    assert normalize_telegram_bot_token(f"bot{TOKEN}") == TOKEN
    assert normalize_telegram_bot_token(f"https://api.telegram.org/bot{TOKEN}/sendMessage") == TOKEN
    assert normalize_telegram_bot_token("") is None
    assert normalize_telegram_bot_token(None) is None


def test_google_news_rss_parse_feed():
    results = GoogleNewsRssSearchProvider.parse_feed(SAMPLE_RSS, max_results=5)
    assert [r.title for r in results] == ["Chip stocks rally as Nvidia surges", "SOX index slips"]
    assert results[0].source == "Reuters"
    assert results[0].published_date.startswith("Wed, 19 Nov 2026")
    assert results[1].snippet == "SOX index slips"


def test_google_news_rss_search_uses_us_locale_and_window():
    provider = GoogleNewsRssSearchProvider()
    fake = MagicMock(status_code=200, text=SAMPLE_RSS)
    with patch("src.search_service.requests.get", return_value=fake) as mocked:
        response = provider.search("SOXL semiconductor", max_results=1, days=3, topic="news")
    assert response.success and len(response.results) == 1
    params = mocked.call_args.kwargs["params"]
    assert params["q"] == "SOXL semiconductor when:3d"
    assert params["gl"] == "US"


def test_google_news_rss_http_error_is_not_success():
    provider = GoogleNewsRssSearchProvider()
    with patch("src.search_service.requests.get", return_value=MagicMock(status_code=503, text="")):
        response = provider.search("x")
    assert not response.success


def test_search_service_registers_rss_only_when_enabled():
    assert not SearchService().is_available
    service = SearchService(google_news_rss_enabled=True)
    assert service.is_available
    assert any(isinstance(p, GoogleNewsRssSearchProvider) for p in service._providers)


def test_ko_dashboard_report_with_fallback_source_does_not_crash():
    from unittest import mock
    from src.analyzer import AnalysisResult
    from src.notification import NotificationService
    from tests.test_notification import _make_config

    with mock.patch("src.notification.get_config",
                    return_value=_make_config(report_renderer_enabled=False, report_language="ko")):
        service = NotificationService()
        result = AnalysisResult(
            code="SOXL",
            name="Direxion Daily Semiconductor Bull 3X Shares",
            sentiment_score=35,
            trend_prediction="强烈看空",
            operation_advice="减仓",
            analysis_summary="반도체 약세",
            report_language="ko",
            market_snapshot={"price": 46.89, "source": "fallback", "pct_chg": -3.2},
        )
        out = service.generate_dashboard_report([result], report_date="2026-11-20")
    assert "SOXL" in out
    assert "대체 소스" in out
    assert "降级兜底" not in out


def test_ko_currency_and_volume_formatting_has_no_chinese_units():
    from src.notification import NotificationService
    from src.analyzer import GeminiAnalyzer

    assert NotificationService._format_per_share(0.093, "USD", "ko") == "0.0930 USD"
    assert NotificationService._format_amount_cn(1.5e9, "USD", "ko") == "1.50B USD"
    # zh path unchanged
    assert NotificationService._format_per_share(0.093, "USD") == "0.0930 美元"
    vol = GeminiAnalyzer._format_volume(object.__new__(GeminiAnalyzer), 1.24e8, "ko")
    assert "亿" not in vol and "shares" in vol
