#!/usr/bin/env python3
import sqlite3
import json
import argparse
from datetime import datetime
import sys

DB_NAME = "checklist.db"


def connect():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS checklist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step INTEGER,
            title TEXT NOT NULL,
            status TEXT DEFAULT 'todo',
            notes TEXT DEFAULT '',
            tags TEXT DEFAULT '',
            updated_at TEXT
        )
    """
    )
    conn.commit()
    conn.close()
    print("✅ Checklist database initialized.")


def seed_checklist():
    steps = [
        (1, "Read project constraints and rubric"),
        (2, "Review project_starter.py and helper functions"),
        (3, "Inspect CSV input/output files"),
        (4, "Design agent architecture (≤5 agents)"),
        (5, "Write agent responsibility descriptions"),
        (6, "Design tool ownership (map helpers to agents)"),
        (7, "Create workflow diagram"),
        (8, "Implement orchestrator agent"),
        (9, "Implement inventory agent"),
        (10, "Implement quote/pricing agent"),
        (11, "Implement sales agent"),
        (12, "Implement finance/reporting agent (optional)"),
        (13, "Wrap starter helper functions as tools"),
        (14, "Implement basic end-to-end happy path"),
        (15, "Add quote intelligence (history + discounts)"),
        (16, "Add inventory validation + reorder logic"),
        (17, "Add delivery date + feasibility checks"),
        (18, "Improve error handling and customer messages"),
        (19, "Run evaluation on sample CSV"),
        (20, "Generate test_results.csv"),
        (21, "Verify rubric success criteria"),
        (22, "Write architecture explanation"),
        (23, "Write evaluation analysis"),
        (24, "Write future improvements section"),
        (25, "Final submission sanity check"),
    ]

    conn = connect()
    cur = conn.cursor()
    for step, title in steps:
        cur.execute(
            """
            INSERT INTO checklist (step, title, updated_at)
            VALUES (?, ?, ?)
        """,
            (step, title, datetime.now().isoformat()),
        )
    conn.commit()
    conn.close()
    print("📋 Checklist seeded with project steps.")


def list_items():
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT id, step, title, status FROM checklist ORDER BY step")
    rows = cur.fetchall()
    conn.close()

    for r in rows:
        print(f"[{r[3]:6}] ({r[1]:02}) #{r[0]} - {r[2]}")


def update_status(item_id, status):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE checklist
        SET status=?, updated_at=?
        WHERE id=?
    """,
        (status, datetime.now().isoformat(), item_id),
    )
    conn.commit()
    conn.close()
    print(f"✔ Item {item_id} marked as {status}.")


def add_note(item_id, note):
    conn = connect()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE checklist
        SET notes=?, updated_at=?
        WHERE id=?
    """,
        (note, datetime.now().isoformat(), item_id),
    )
    conn.commit()
    conn.close()
    print(f"📝 Note added to item {item_id}.")


def export_json(filename):
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM checklist ORDER BY step")
    rows = cur.fetchall()
    cols = [c[0] for c in cur.description]
    conn.close()

    data = [dict(zip(cols, r)) for r in rows]
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"📤 Checklist exported to {filename}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "init",
            "seed",
            "list",
            "done",
            "doing",
            "todo",
            "blocked",
            "note",
            "export",
            "rename",
        ],
    )
    parser.add_argument("arg1", nargs="?")
    parser.add_argument("arg2", nargs="?")

    args = parser.parse_args()

    if args.command == "init":
        init_db()
    elif args.command == "seed":
        seed_checklist()
    elif args.command == "list":
        list_items()
    elif args.command in ["done", "doing", "todo", "blocked"]:
        update_status(int(args.arg1), args.command)
    elif args.command == "note":
        add_note(int(args.arg1), args.arg2)
    elif args.command == "export":
        export_json(args.arg1)
    elif args.command == "rename":
        item_id = int(sys.argv[2])
        new_title = " ".join(sys.argv[3:])
        rename_item(item_id, new_title)


def rename_item(item_id, new_title):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("UPDATE checklist SET title = ? WHERE id = ?", (new_title, item_id))
    conn.commit()
    conn.close()
    print(f"✏️ Item #{item_id} renamed.")


if __name__ == "__main__":
    main()
