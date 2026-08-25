"""
db_builder.py - SQLite Database Builder with Normalized Schema and FTS5 Full-Text Index
"""

import sqlite3
import json
from typing import List, Dict, Any

def create_database(db_path: str = "uyghur_english_dictionary.sqlite") -> sqlite3.Connection:
    """Create normalized SQLite database schema with full-text search indexes."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA foreign_keys = ON;")

    # Drop old tables if rebuilding
    cur.execute("DROP TABLE IF EXISTS entries_fts;")
    cur.execute("DROP TABLE IF EXISTS correspondences;")
    cur.execute("DROP TABLE IF EXISTS etymologies;")
    cur.execute("DROP TABLE IF EXISTS cross_references;")
    cur.execute("DROP TABLE IF EXISTS special_meanings;")
    cur.execute("DROP TABLE IF EXISTS sayings;")
    cur.execute("DROP TABLE IF EXISTS examples;")
    cur.execute("DROP TABLE IF EXISTS senses;")
    cur.execute("DROP TABLE IF EXISTS entries;")

    # Main entries table
    cur.execute("""
    CREATE TABLE entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        headword_schwarz TEXT NOT NULL,
        headword_uly TEXT NOT NULL,
        headword_uey TEXT NOT NULL,
        headword_uyy TEXT,
        headword_ipa TEXT,
        homograph_number TEXT,
        pos TEXT,
        domain TEXT,
        page_book INTEGER NOT NULL,
        page_pdf INTEGER NOT NULL,
        column INTEGER NOT NULL,
        raw_body TEXT,
        json_data TEXT
    );
    """)

    # Senses table
    cur.execute("""
    CREATE TABLE senses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id INTEGER NOT NULL,
        sense_number INTEGER DEFAULT 1,
        pos TEXT,
        voice TEXT,
        domain TEXT,
        definition TEXT NOT NULL,
        raw_text TEXT,
        FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE
    );
    """)

    # Examples table
    cur.execute("""
    CREATE TABLE examples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sense_id INTEGER NOT NULL,
        uyghur_uey TEXT NOT NULL,
        uyghur_uly TEXT NOT NULL,
        uyghur_schwarz TEXT NOT NULL,
        english TEXT,
        domain TEXT,
        FOREIGN KEY (sense_id) REFERENCES senses(id) ON DELETE CASCADE
    );
    """)

    # Sayings / Proverbs table
    cur.execute("""
    CREATE TABLE sayings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sense_id INTEGER NOT NULL,
        uyghur_uey TEXT NOT NULL,
        uyghur_uly TEXT NOT NULL,
        uyghur_schwarz TEXT NOT NULL,
        english TEXT,
        explanation TEXT,
        FOREIGN KEY (sense_id) REFERENCES senses(id) ON DELETE CASCADE
    );
    """)

    # Special Meanings / Idioms table
    cur.execute("""
    CREATE TABLE special_meanings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sense_id INTEGER NOT NULL,
        uyghur_uey TEXT NOT NULL,
        uyghur_uly TEXT NOT NULL,
        uyghur_schwarz TEXT NOT NULL,
        english TEXT,
        FOREIGN KEY (sense_id) REFERENCES senses(id) ON DELETE CASCADE
    );
    """)

    # Cross References table
    cur.execute("""
    CREATE TABLE cross_references (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sense_id INTEGER NOT NULL,
        ref_type TEXT NOT NULL,
        target_schwarz TEXT NOT NULL,
        target_uey TEXT NOT NULL,
        target_uly TEXT NOT NULL,
        FOREIGN KEY (sense_id) REFERENCES senses(id) ON DELETE CASCADE
    );
    """)

    # Etymologies table
    cur.execute("""
    CREATE TABLE etymologies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sense_id INTEGER NOT NULL,
        donor_language TEXT,
        lang_code TEXT,
        word TEXT,
        ipa TEXT,
        raw_text TEXT,
        FOREIGN KEY (sense_id) REFERENCES senses(id) ON DELETE CASCADE
    );
    """)

    # Correspondences table (14 Turkic/Mongolic cognates)
    cur.execute("""
    CREATE TABLE correspondences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sense_id INTEGER NOT NULL,
        language TEXT NOT NULL,
        lang_code TEXT NOT NULL,
        word TEXT NOT NULL,
        ipa TEXT,
        note TEXT,
        FOREIGN KEY (sense_id) REFERENCES senses(id) ON DELETE CASCADE
    );
    """)

    # B-Tree Indexes
    cur.execute("CREATE INDEX idx_headword_schwarz ON entries(headword_schwarz);")
    cur.execute("CREATE INDEX idx_headword_uly ON entries(headword_uly);")
    cur.execute("CREATE INDEX idx_headword_uey ON entries(headword_uey);")
    cur.execute("CREATE INDEX idx_page_book ON entries(page_book);")

    # FTS5 Full-Text Search Virtual Table
    cur.execute("""
    CREATE VIRTUAL TABLE entries_fts USING fts5(
        entry_id UNINDEXED,
        headword_uey,
        headword_uly,
        headword_schwarz,
        headword_uyy,
        definitions,
        examples_uey,
        examples_english,
        sayings_uey,
        sayings_english,
        special_meanings_uey,
        special_meanings_english,
        raw_body
    );
    """)

    conn.commit()
    return conn


def insert_entry(conn: sqlite3.Connection, entry: Dict[str, Any]) -> int:
    """Insert a single structured entry and its children into SQLite database."""
    cur = conn.cursor()
    
    hw = entry["headword"]
    pos_str = ', '.join(entry.get("pos", [])) if entry.get("pos") else None
    domain_str = ', '.join(entry.get("domain", [])) if entry.get("domain") else None
    json_str = json.dumps(entry, ensure_ascii=False)

    cur.execute("""
    INSERT INTO entries (
        headword_schwarz, headword_uly, headword_uey, headword_uyy, headword_ipa,
        homograph_number, pos, domain, page_book, page_pdf, column, raw_body, json_data
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        hw["schwarz"],
        hw["uly"],
        hw["uey"],
        hw.get("uyy"),
        hw.get("ipa"),
        entry.get("homograph_number"),
        pos_str,
        domain_str,
        entry["page_book"],
        entry["page_pdf"],
        entry["column"],
        entry.get("raw_body", ""),
        json_str
    ))
    entry_id = cur.lastrowid

    all_defs = []
    all_ex_uey = []
    all_ex_en = []
    all_say_uey = []
    all_say_en = []
    all_sm_uey = []
    all_sm_en = []

    for s in entry.get("senses", []):
        cur.execute("""
        INSERT INTO senses (entry_id, sense_number, pos, voice, domain, definition, raw_text)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            entry_id,
            s.get("sense_number", 1),
            s.get("pos"),
            s.get("voice"),
            s.get("domain"),
            s.get("definition", ""),
            s.get("raw_text", "")
        ))
        sense_id = cur.lastrowid
        all_defs.append(s.get("definition", ""))

        # Insert examples
        for ex in s.get("examples", []):
            cur.execute("""
            INSERT INTO examples (sense_id, uyghur_uey, uyghur_uly, uyghur_schwarz, english, domain)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (
                sense_id,
                ex["uyghur_uey"],
                ex["uyghur_uly"],
                ex["uyghur_schwarz"],
                ex.get("english", ""),
                ex.get("domain")
            ))
            all_ex_uey.append(ex["uyghur_uey"])
            all_ex_en.append(ex.get("english", ""))

        # Insert sayings
        for say in s.get("sayings", []):
            cur.execute("""
            INSERT INTO sayings (sense_id, uyghur_uey, uyghur_uly, uyghur_schwarz, english, explanation)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (
                sense_id,
                say["uyghur_uey"],
                say["uyghur_uly"],
                say["uyghur_schwarz"],
                say.get("english", ""),
                say.get("explanation")
            ))
            all_say_uey.append(say["uyghur_uey"])
            all_say_en.append(say.get("english", ""))

        # Insert special meanings
        for sm in s.get("special_meanings", []):
            cur.execute("""
            INSERT INTO special_meanings (sense_id, uyghur_uey, uyghur_uly, uyghur_schwarz, english)
            VALUES (?, ?, ?, ?, ?);
            """, (
                sense_id,
                sm["uyghur_uey"],
                sm["uyghur_uly"],
                sm["uyghur_schwarz"],
                sm.get("english", "")
            ))
            all_sm_uey.append(sm["uyghur_uey"])
            all_sm_en.append(sm.get("english", ""))

        # Insert cross references
        for cr in s.get("cross_references", []):
            cur.execute("""
            INSERT INTO cross_references (sense_id, ref_type, target_schwarz, target_uey, target_uly)
            VALUES (?, ?, ?, ?, ?);
            """, (
                sense_id,
                cr["type"],
                cr["target_schwarz"],
                cr["target_uey"],
                cr["target_uly"]
            ))

        # Insert etymologies
        for et in s.get("etymologies", []):
            chain = et.get("chain", [])
            if chain:
                for item in chain:
                    cur.execute("""
                    INSERT INTO etymologies (sense_id, donor_language, lang_code, word, ipa, raw_text)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """, (
                        sense_id,
                        item.get("language"),
                        item.get("lang_code"),
                        item.get("word"),
                        item.get("ipa"),
                        et.get("raw")
                    ))
            else:
                cur.execute("""
                INSERT INTO etymologies (sense_id, donor_language, lang_code, word, ipa, raw_text)
                VALUES (?, ?, ?, ?, ?, ?);
                """, (sense_id, None, None, None, None, et.get("raw")))

        # Insert correspondences
        for co in s.get("correspondences", []):
            cur.execute("""
            INSERT INTO correspondences (sense_id, language, lang_code, word, ipa, note)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (
                sense_id,
                co["language"],
                co["lang_code"],
                co["word"],
                co.get("ipa"),
                co.get("note")
            ))

    # Insert FTS index record
    cur.execute("""
    INSERT INTO entries_fts (
        entry_id, headword_uey, headword_uly, headword_schwarz, headword_uyy,
        definitions, examples_uey, examples_english, sayings_uey, sayings_english,
        special_meanings_uey, special_meanings_english, raw_body
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        entry_id,
        hw["uey"],
        hw["uly"],
        hw["schwarz"],
        hw.get("uyy", ""),
        ' '.join(all_defs),
        ' '.join(all_ex_uey),
        ' '.join(all_ex_en),
        ' '.join(all_say_uey),
        ' '.join(all_say_en),
        ' '.join(all_sm_uey),
        ' '.join(all_sm_en),
        entry.get("raw_body", "")
    ))

    return entry_id


def save_dictionary_to_db_and_json(
    entries: List[Dict[str, Any]],
    db_path: str = "uyghur_english_dictionary.sqlite",
    json_path: str = "uyghur_english_dictionary.json"
):
    """Save a list of parsed entries to both SQLite database and JSON file."""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    conn = create_database(db_path)
    for entry in entries:
        insert_entry(conn, entry)
    conn.commit()
    conn.close()
