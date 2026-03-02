"""Daily 10-Year U.S. Treasury yield monitor with email alert."""

import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

import yaml
import yfinance as yf

CONFIG_PATH = Path(__file__).parent / "config.yaml"
DEFAULT_TICKER = "^TNX"  # CBOE 10-Year Treasury Note Yield Index (value = % yield)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and return configuration from a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_yield(ticker: str = DEFAULT_TICKER) -> float:
    """Fetch the most recent closing yield for the given ticker symbol."""
    data = yf.Ticker(ticker)
    history = data.history(period="5d")
    if history.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    return float(history["Close"].iloc[-1])


def build_email(yield_pct: float, threshold: float, email_cfg: dict) -> EmailMessage:
    """Construct the alert EmailMessage object."""
    msg = EmailMessage()
    msg["Subject"] = (
        f"Treasury Alert: 10Y yield {yield_pct:.2f}% is below {threshold:.2f}%"
    )
    msg["From"] = email_cfg["sender"]
    msg["To"] = email_cfg["recipient"]
    msg.set_content(
        f"The U.S. 10-Year Treasury yield has fallen below your threshold.\n\n"
        f"  Current yield : {yield_pct:.2f}%\n"
        f"  Threshold     : {threshold:.2f}%\n\n"
        f"Review your positions accordingly."
    )
    return msg


def send_alert(yield_pct: float, threshold: float, email_cfg: dict) -> None:
    """Send an email alert via SMTP using credentials from SMTP_PASSWORD env var."""
    password = os.environ.get("SMTP_PASSWORD")
    if not password:
        raise EnvironmentError(
            "SMTP_PASSWORD environment variable is not set. "
            "Set it before running: export SMTP_PASSWORD='your-app-password'"
        )
    msg = build_email(yield_pct, threshold, email_cfg)
    context = ssl.create_default_context()
    with smtplib.SMTP(email_cfg["smtp_host"], email_cfg["smtp_port"]) as server:
        server.starttls(context=context)
        server.login(email_cfg["sender"], password)
        server.send_message(msg)
    print(f"Alert sent: yield {yield_pct:.2f}% < threshold {threshold:.2f}%")


def main() -> None:
    """Fetch today's 10Y Treasury yield and send an alert if it's below threshold."""
    cfg = load_config()
    threshold = float(cfg["treasury"]["threshold"])
    ticker = cfg["treasury"].get("ticker", DEFAULT_TICKER)

    yield_pct = fetch_yield(ticker)
    print(f"Current 10Y Treasury yield: {yield_pct:.2f}%")

    if yield_pct < threshold:
        send_alert(yield_pct, threshold, cfg["email"])
    else:
        print(f"Yield {yield_pct:.2f}% >= threshold {threshold:.2f}%. No alert sent.")


if __name__ == "__main__":
    main()
