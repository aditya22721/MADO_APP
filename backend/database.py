import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "mediguide.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            phone TEXT UNIQUE,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS emergency_contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            phone TEXT,
            email TEXT,
            carrier TEXT,
            relationship TEXT,
            is_primary INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    try:
        c.execute("ALTER TABLE emergency_contacts ADD COLUMN email TEXT")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE emergency_contacts ADD COLUMN carrier TEXT")
    except Exception:
        pass

    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT,
            response TEXT,
            is_emergency INTEGER DEFAULT 0,
            location TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS emergency_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            emergency_type TEXT,
            location TEXT,
            message TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅ Database ready at {DB_PATH}")


def add_contact(user_id, name, phone, relationship, is_primary=False,
                email=None, carrier=None):
    conn = get_db()
    c = conn.cursor()
    if is_primary:
        c.execute("UPDATE emergency_contacts SET is_primary=0 WHERE user_id=?", (user_id,))
    c.execute(
        """INSERT INTO emergency_contacts
           (user_id, name, phone, email, relationship, is_primary, carrier)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, name, phone, email, relationship,
         1 if is_primary else 0, carrier),
    )
    conn.commit()
    cid = c.lastrowid
    conn.close()
    return cid


def get_contacts(user_id):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM emergency_contacts WHERE user_id=? ORDER BY is_primary DESC",
        (user_id,),
    )
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_contact(contact_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM emergency_contacts WHERE id=?", (contact_id,))
    conn.commit()
    ok = c.rowcount > 0
    conn.close()
    return ok


def save_chat(user_id, message, response, is_emergency=False, location=None):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """INSERT INTO chat_history
           (user_id, message, response, is_emergency, location)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, message, response, 1 if is_emergency else 0,
         json.dumps(location) if location else None),
    )
    conn.commit()
    conn.close()


def get_chat_history(user_id, limit=20):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM chat_history WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("location"):
            d["location"] = json.loads(d["location"])
        out.append(d)
    return out


def save_emergency_alert(user_id, emergency_type, message, location=None, status="active"):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """INSERT INTO emergency_alerts
           (user_id, emergency_type, location, message, status)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, emergency_type,
         json.dumps(location) if location else None,
         message, status),
    )
    conn.commit()
    aid = c.lastrowid
    conn.close()
    return aid


def get_emergency_alerts(user_id, limit=10):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM emergency_alerts WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        if d.get("location"):
            d["location"] = json.loads(d["location"])
        out.append(d)
    return out


def update_alert_status(alert_id, status):
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE emergency_alerts SET status=? WHERE id=?", (status, alert_id))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()