"""Tests for treasury_monitor.py."""

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from treasury_monitor import build_email, fetch_yield, load_config, send_alert

SAMPLE_CONFIG = {
    "treasury": {"threshold": 4.0, "ticker": "^TNX"},
    "email": {
        "sender": "sender@example.com",
        "recipient": "recipient@example.com",
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
    },
}


def test_load_config(tmp_path):
    """Config file is parsed and values are accessible."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(SAMPLE_CONFIG))
    result = load_config(cfg_file)
    assert result["treasury"]["threshold"] == 4.0
    assert result["email"]["smtp_port"] == 587


def test_fetch_yield_returns_float():
    """fetch_yield returns the latest close as a float."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({"Close": [4.25, 4.30]})
    with patch("treasury_monitor.yf.Ticker", return_value=mock_ticker):
        result = fetch_yield("^TNX")
    assert isinstance(result, float)
    assert result == pytest.approx(4.30)


def test_fetch_yield_raises_on_empty_history():
    """fetch_yield raises ValueError when the ticker returns no data."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    with patch("treasury_monitor.yf.Ticker", return_value=mock_ticker):
        with pytest.raises(ValueError, match="No data returned"):
            fetch_yield("^TNX")


def test_build_email_contains_yield_and_threshold():
    """Email subject and body contain the yield and threshold values."""
    msg = build_email(3.85, 4.0, SAMPLE_CONFIG["email"])
    assert "3.85" in msg["Subject"]
    assert "4.00" in msg["Subject"]
    assert msg["To"] == "recipient@example.com"
    assert "3.85" in msg.get_payload()
    assert "4.00" in msg.get_payload()


def test_send_alert_raises_without_smtp_password():
    """send_alert raises EnvironmentError when SMTP_PASSWORD is not set."""
    clean_env = {k: v for k, v in os.environ.items() if k != "SMTP_PASSWORD"}
    with patch.dict(os.environ, clean_env, clear=True):
        with pytest.raises(EnvironmentError, match="SMTP_PASSWORD"):
            send_alert(3.85, 4.0, SAMPLE_CONFIG["email"])
