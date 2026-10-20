
import os
import time
import threading
import requests
from dotenv import load_dotenv
import db_manager

load_dotenv()  # reads .env in the project root

# ── Fast2SMS config (from environment) ──────────────────────────────────────
FAST2SMS_API_KEY = os.environ["FAST2SMS_API_KEY"]
FAST2SMS_URL      = "https://www.fast2sms.com/dev/bulkV2"

REMINDER_DELAY_SECONDS = 5

# ── Risk levels that trigger SMS ────────────────────────────────────────────
ALERT_LEVELS = {"Moderate", "High", "Extreme"}

# ── SMS message templates ───────────────────────────────────────────────────
_MESSAGES = {
    "Extreme": (
        "FLOOD EMERGENCY - {district}\n"
        "Risk Level: EXTREME\n"
        "Move to higher ground IMMEDIATELY.\n"
        "- India Flood AI Guardian"
    ),
    "High": (
        "FLOOD ALERT - {district}\n"
        "Risk Level: HIGH\n"
        "Stay alert. Avoid low-lying areas.\n"
        "- India Flood AI Guardian"
    ),
    "Moderate": (
        "FLOOD ADVISORY - {district}\n"
        "Risk Level: MODERATE\n"
        "Monitor updates and be prepared.\n"
        "- India Flood AI Guardian"
    ),
}

_REMINDER_PREFIX = "REMINDER:\n"


def _to_fast2sms_number(phone: str) -> str:
    """
    Fast2SMS Quick SMS expects a bare 10-digit Indian mobile number,
    no country code, no '+'. Our DB stores E.164 like '+918897988139'.
    """
    digits = "".join(ch for ch in phone if ch.isdigit())
    # Keep last 10 digits — strips leading '91' country code if present
    return digits[-10:]


def _fast2sms_send(body: str, to: str) -> dict:
    """
    Send one SMS via Fast2SMS Quick SMS route. Retries once on failure.
    Returns {"sent": True/False, "request_id": ..., "error": ...}
    """
    number = _to_fast2sms_number(to)
    headers = {
        "authorization": FAST2SMS_API_KEY,
        "Content-Type": "application/x-www-form-urlencoded",
        "Cache-Control": "no-cache",
    }
    payload = {
        "route": "q",          # Quick SMS — custom text, no DLT needed
        "message": body,
        "language": "english",
        "flash": 0,
        "numbers": number,
    }

    last_error = ""
    for attempt in range(2):
        try:
            r = requests.post(FAST2SMS_URL, headers=headers, data=payload, timeout=15)
            data = r.json()
            if data.get("return") is True:
                return {"sent": True, "request_id": data.get("request_id", "")}
            else:
                last_error = f"[attempt {attempt+1}/2] Fast2SMS: {data.get('message', data)}"
        except Exception as e:
            last_error = f"[attempt {attempt+1}/2] {type(e).__name__}: {str(e)[:200]}"

        if attempt == 0:
            time.sleep(2)

    return {"sent": False, "error": last_error}


def _send_reminder(user: dict, district: str, risk: str, window_key: str) -> None:
    """
    Daemon thread: waits REMINDER_DELAY_SECONDS then sends reminder SMS.
    Re-checks dedup before sending — safe against duplicate threads.
    """
    time.sleep(REMINDER_DELAY_SECONDS)

    if not db_manager.should_send(user["phone"], district, risk, "reminder", window_key):
        return

    body = _REMINDER_PREFIX + _MESSAGES[risk].format(district=district)
    result = _fast2sms_send(body, user["phone"])

    db_manager.log_alert(
        user_id=user["id"],
        district=district,
        phone=user["phone"],
        risk_level=risk,
        alert_type="reminder",
        window_key=window_key,
        sent=result.get("sent", False),
        error_msg=result.get("error", ""),
    )


def process_all_alerts(df_predictions, window_key: str) -> list:
    """
    Main entry point — called by app.py once per 30-min window.

    Args:
        df_predictions: DataFrame with columns [district, knn_risk]
        window_key: string like '20260506_0700' — identifies this 30-min window

    Returns:
        list of {"district": ..., "risk": ..., "sent": bool, "error": str}
    """
    live_risk_map = {
        row["district"].strip().lower(): row["knn_risk"]
        for _, row in df_predictions.iterrows()
    }

    users_at_risk = db_manager.get_users_at_risk(live_risk_map)

    results = []
    for user in users_at_risk:
        district = user["district"]
        risk = user["risk"]

        if not db_manager.should_send(user["phone"], district, risk, "initial", window_key):
            continue

        body = _MESSAGES[risk].format(district=district)
        result = _fast2sms_send(body, user["phone"])

        db_manager.log_alert(
            user_id=user["id"],
            district=district,
            phone=user["phone"],
            risk_level=risk,
            alert_type="initial",
            window_key=window_key,
            sent=result.get("sent", False),
            error_msg=result.get("error", ""),
        )

        if result.get("sent"):
            threading.Thread(
                target=_send_reminder,
                args=(user, district, risk, window_key),
                daemon=True,
            ).start()

        results.append({
            "district": district,
            "risk": risk,
            "sent": result.get("sent", False),
            "error": result.get("error", ""),
        })

    return results