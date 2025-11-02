import sqlite3
from datetime import datetime

DB_NAME = 'requirements.db'

def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requirements_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            upload_date TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requirements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            requirement_text TEXT NOT NULL,
            is_functional INTEGER NOT NULL,
            is_non_functional INTEGER NOT NULL,
            is_ambiguous INTEGER NOT NULL,
            is_measurable INTEGER NOT NULL,
            notes TEXT,
            file_id INTEGER,
            FOREIGN KEY (file_id) REFERENCES requirements_files(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database and tables created successfully.")

def insert_requirement(requirement_text, is_functional, is_ambiguous,
                       is_measurable, notes, file_name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM requirements_files WHERE file_name = ?", (file_name,))
    result = cursor.fetchone()
    if result:
        file_id = result[0]
    else:
        upload_date = datetime.now().strftime("%Y-%m-%d")
        cursor.execute('''
            INSERT INTO requirements_files (file_name, upload_date)
            VALUES (?, ?)
        ''', (file_name, upload_date))
        file_id = cursor.lastrowid

    cursor.execute('''
        INSERT INTO requirements (
            requirement_text,
            is_functional,
            is_non_functional,
            is_ambiguous,
            is_measurable,
            notes,
            file_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        requirement_text,
        1 if is_functional else 0,
        0 if is_functional else 1,
        1 if is_ambiguous else 0,
        1 if is_measurable else 0,
        notes,
        file_id
    ))

    conn.commit()
    conn.close()
    print(f"✅ Inserted: {requirement_text[:40]}...")

def get_requirements():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT r.id, r.requirement_text, r.is_functional, r.is_non_functional,
               r.is_ambiguous, r.is_measurable, r.notes, f.file_name
        FROM requirements r
        JOIN requirements_files f ON r.file_id = f.id
    ''')

    rows = cursor.fetchall()
    conn.close()

    requirements = []
    for row in rows:
        requirements.append({
            "id": row[0],
            "text": row[1],
            "functional": bool(row[2]),
            "non_functional": bool(row[3]),
            "ambiguous": bool(row[4]),
            "measurable": bool(row[5]),
            "notes": row[6],
            "file": row[7]
        })
    return requirements