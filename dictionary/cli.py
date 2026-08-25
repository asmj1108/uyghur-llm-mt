"""
cli.py - Command Line Interface for Uyghur-English Dictionary Digitalization
"""

import argparse
import sys
import os
import sqlite3
import json

import pipeline
import converter

def main():
    parser = argparse.ArgumentParser(description="Uyghur-English Dictionary Digitalization CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: process
    proc_parser = subparsers.add_parser("process", help="Process a range of pages")
    proc_parser.add_argument("--start", type=int, default=26, help="Start PDF page (default: 26)")
    proc_parser.add_argument("--end", type=int, default=30, help="End PDF page (default: 30)")
    proc_parser.add_argument("--output", type=str, default="uyghur_dict_output", help="Output prefix")
    proc_parser.add_argument("--crops-dir", type=str, default="crops", help="Crops directory")
    proc_parser.add_argument("--cache-dir", type=str, default="transcription_cache", help="Cache directory")

    # Command: search
    search_parser = subparsers.add_parser("search", help="Search the SQLite database via FTS")
    search_parser.add_argument("query", type=str, help="Search query (Arabic, ULY, Latin, or English)")
    search_parser.add_argument("--db", type=str, default="staged_test.sqlite", help="Database path")
    search_parser.add_argument("--limit", type=int, default=10, help="Maximum results")

    # Command: convert
    conv_parser = subparsers.add_parser("convert", help="Convert text between Uyghur scripts")
    conv_parser.add_argument("text", type=str, help="Schwarz Latin text to convert")

    # Command: stats
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument("--db", type=str, default="staged_test.sqlite", help="Database path")

    args = parser.parse_args()

    if args.command == "process":
        print(f"Processing PDF pages {args.start} to {args.end}...")
        entries = pipeline.process_page_range(
            start_page_pdf=args.start,
            end_page_pdf=args.end,
            crops_dir=args.crops_dir,
            cache_dir=args.cache_dir,
            output_prefix=args.output
        )
        print(f"Processed {len(entries)} entries. Saved to {args.output}.json and {args.output}.sqlite.")

    elif args.command == "search":
        if not os.path.exists(args.db):
            print(f"Database {args.db} not found.")
            sys.exit(1)
        conn = sqlite3.connect(args.db)
        cur = conn.cursor()
        
        cur.execute("""
        SELECT e.id, e.headword_schwarz, e.headword_uly, e.headword_uey, e.homograph_number, e.pos, e.page_book, s.definition
        FROM entries_fts f
        JOIN entries e ON f.entry_id = e.id
        LEFT JOIN senses s ON s.entry_id = e.id
        WHERE entries_fts MATCH ?
        LIMIT ?;
        """, (f"{args.query}*", args.limit))
        
        rows = cur.fetchall()
        if not rows:
            print(f"No results found for '{args.query}'.")
        else:
            print(f"Found {len(rows)} results for '{args.query}':\n")
            for r in rows:
                homo = f" {r[4]}" if r[4] else ""
                print(f"• #{r[0]} {r[1]}{homo} [ULY: {r[2]} | UEY: {r[3]}] ({r[5] or 'N/A'}) - Book p. {r[6]}")
                print(f"  Definition: {r[7]}\n")

    elif args.command == "convert":
        print(f"Original: {args.text}")
        print(f"ULY:      {converter.custom_to_uly(args.text)}")
        print(f"UEY (Ar): {converter.custom_to_arabic(args.text)}")
        print(f"UYY:      {converter.custom_to_uyy(args.text)}")
        print(f"IPA:      {converter.custom_to_ipa(args.text)}")

    elif args.command == "stats":
        if not os.path.exists(args.db):
            print(f"Database {args.db} not found.")
            sys.exit(1)
        conn = sqlite3.connect(args.db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM entries;")
        n_entries = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM senses;")
        n_senses = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM examples;")
        n_ex = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM sayings;")
        n_say = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM special_meanings;")
        n_sm = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM cross_references;")
        n_refs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM etymologies;")
        n_etym = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM correspondences;")
        n_corr = cur.fetchone()[0]
        print(f"Database: {args.db}")
        print(f"• Total Headwords / Entries: {n_entries}")
        print(f"• Total Senses:             {n_senses}")
        print(f"• Examples (with UEY & ULY): {n_ex}")
        print(f"• Sayings / Proverbs (¶):   {n_say}")
        print(f"• Special Meanings (♦):     {n_sm}")
        print(f"• Cross References:         {n_refs}")
        print(f"• Etymologies (<):          {n_etym}")
        print(f"• Correspondences (Cognates): {n_corr}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
