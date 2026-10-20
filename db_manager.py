
import sqlite3, re, os

BASE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "flood_guardian.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name     TEXT    NOT NULL,
    phone    TEXT    NOT NULL UNIQUE,
    district TEXT    NOT NULL,
    active   INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS alert_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    district   TEXT    NOT NULL,
    phone      TEXT    NOT NULL,
    risk_level TEXT    NOT NULL,
    alert_type TEXT    NOT NULL DEFAULT 'initial',
    window_key TEXT    NOT NULL,
    sent       INTEGER NOT NULL DEFAULT 0,
    error_msg  TEXT    DEFAULT '',
    sent_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_dedup
    ON alert_log(phone, district, risk_level, alert_type, window_key);
"""

# ── Edit these to match your districts.csv spelling exactly ───────────────────
_SEED_USERS = [
    ("My Account",    "+91XXXXXXXXXX", "Visakhapatnam"),
    
]

# Risk levels that trigger SMS — Low is always excluded
_ALERT_LEVELS = {"Moderate", "High", "Extreme"}


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """
    Create tables + FULL SYNC with _SEED_USERS.
    UPSERT: if phone already exists → update name+district so changes apply on restart.
    DEACTIVATE: any active user whose phone is NOT in the current _SEED_USERS
    list gets set to active=0. Without this, changing a phone number in
    _SEED_USERS just ADDS a new user on top of the old one — the old number
    stays active forever and keeps receiving alerts alongside the new one.
    """
    c = _conn()
    c.executescript(_SCHEMA)
    c.commit()

    seed_phones = [phone for _, phone, _ in _SEED_USERS]

    for name, phone, district in _SEED_USERS:
        exists = c.execute("SELECT id FROM users WHERE phone=?", (phone,)).fetchone()
        if exists:
            # Always sync — ensures district edits in _SEED_USERS take effect
            c.execute(
                "UPDATE users SET name=?, district=?, active=1 WHERE phone=?",
                (name, district, phone)
            )
        else:
            c.execute(
                "INSERT INTO users (name, phone, district) VALUES (?,?,?)",
                (name, phone, district)
            )

    # Deactivate any previously-seeded number that's no longer in the list —
    # this is what actually stops old numbers from still receiving alerts.
    if seed_phones:
        placeholders = ",".join("?" * len(seed_phones))
        c.execute(
            f"UPDATE users SET active=0 WHERE phone NOT IN ({placeholders}) AND active=1",
            seed_phones
        )
    c.commit()
    c.close()


def get_users_at_risk(live_risk_map: dict) -> list:
    """
    SQL-level filter — Standard Practice per guide requirement.
    Equivalent to: SELECT * FROM users WHERE district_risk >= 1 (Moderate/High/Extreme).

    Args:
        live_risk_map: {district_name_lowercase: risk_label} from KNN output

    Returns:
        list of {id, name, phone, district, risk} — only Moderate/High/Extreme users.
        Low-risk users are never returned — no SMS sent to them.
    """
    # Filter at Python dict level to build the SQL IN clause — only alert levels
    at_risk_districts = {
        dist for dist, risk in live_risk_map.items()
        if risk in _ALERT_LEVELS
    }
    if not at_risk_districts:
        return []  # No at-risk districts — send nothing

    # SQL does the real filtering — memory efficient, no full table scan in Python
    placeholders = ",".join("?" * len(at_risk_districts))
    sql = f"""
        SELECT id, name, phone, district
        FROM   users
        WHERE  active = 1
          AND  LOWER(district) IN ({placeholders})
    """
    c = _conn()
    rows = c.execute(sql, list(at_risk_districts)).fetchall()
    c.close()

    result = []
    for row in rows:
        risk = live_risk_map.get(row["district"].strip().lower(), "Low")
        result.append({
            "id":       row["id"],
            "name":     row["name"],
            "phone":    row["phone"],
            "district": row["district"],
            "risk":     risk,
        })
    return result


def should_send(phone: str, district: str, risk_level: str,
                alert_type: str, window_key: str) -> bool:
    """True if this alert was NOT already successfully sent in this 30-min window."""
    c = _conn()
    row = c.execute(
        "SELECT id FROM alert_log "
        "WHERE phone=? AND district=? AND risk_level=? "
        "  AND alert_type=? AND window_key=? AND sent=1",
        (phone, district, risk_level, alert_type, window_key)
    ).fetchone()
    c.close()
    return row is None  # True = not sent yet = go ahead


def log_alert(user_id: int, district: str, phone: str, risk_level: str,
              alert_type: str, window_key: str, sent: bool, error_msg: str = "") -> None:
    """Write one row to alert_log — called after every SMS attempt."""
    c = _conn()
    c.execute(
        "INSERT INTO alert_log "
        "(user_id, district, phone, risk_level, alert_type, window_key, sent, error_msg)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (user_id, district, phone, risk_level, alert_type, window_key, int(sent), error_msg)
    )
    c.commit()
    c.close()


def get_all_users() -> list:
    """Return active users — used ONLY for map marker highlighting (no UI display)."""
    c = _conn()
    rows = c.execute(
        "SELECT id, district FROM users WHERE active=1 ORDER BY id"
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


def add_user(name: str, phone: str, district: str) -> dict:
    if not re.match(r"^\+\d{10,15}$", phone):
        return {"ok": False, "error": f"Invalid phone format: {phone}"}
    c = _conn()
    try:
        cur = c.execute(
            "INSERT INTO users (name, phone, district) VALUES (?,?,?)",
            (name, phone, district)
        )
        c.commit()
        return {"ok": True, "id": cur.lastrowid}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": f"Phone {phone} already registered."}
    finally:
        c.close()


def deactivate_user(user_id: int) -> None:
    c = _conn()
    c.execute("UPDATE users SET active=0 WHERE id=?", (user_id,))
    c.commit()
    c.close()


def clear_alert_log() -> None:
    """Wipe alert_log to reset dedup. Use for testing only."""
    c = _conn()
    c.execute("DELETE FROM alert_log")
    c.commit()
    c.close()
    print("✅ alert_log cleared — dedup reset.")


if __name__ == "__main__":
    init_db()
    print(f"✅ DB: {DB_PATH}")
    c = _conn()
    users = c.execute("SELECT * FROM users WHERE active=1").fetchall()
    print(f"\nActive users ({len(users)}):")
    for u in users:
        print(f"  ID={u['id']}  name={u['name']}  district={u['district']}  phone={u['phone']}")
    last = c.execute("SELECT * FROM alert_log ORDER BY id DESC LIMIT 5").fetchall()
    print(f"\nLast {len(last)} alerts:")
    for a in last:
        print(f"  {a['sent_at']}  {a['district']}  {a['risk_level']}  "
              f"{a['alert_type']}  sent={a['sent']}  err={str(a['error_msg'])[:80]}")
    c.close()