"""
cli.py - Command Line Interface for Uyghur-English Dictionary Digitalization
"""

import argparse
import sys
import os
import sqlite3

import cropper
import pipeline
import converter

def main():
    parser = argparse.ArgumentParser(description="Uyghur-English Dictionary Digitalization CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: batch (End-to-end parallel batch pipeline)
    batch_parser = subparsers.add_parser("batch", help="End-to-end concurrent batch processing (crop -> transcribe -> build)")
    batch_parser.add_argument("--pdf", type=str, default="An_Uyghur_English_Dictionary_Complete.pdf", help="Source PDF file")
    batch_parser.add_argument("--start", type=int, default=26, help="Start PDF page (default: 26)")
    batch_parser.add_argument("--end", type=int, default=972, help="End PDF page (default: 972)")
    batch_parser.add_argument("--output", type=str, default="uyghur_english_dictionary", help="Output prefix for .jsonl and .sqlite")
    batch_parser.add_argument("--crops-dir", type=str, default="crops", help="Crops directory")
    batch_parser.add_argument("--cache-dir", type=str, default="transcription_cache", help="Cache directory")
    batch_parser.add_argument("--crop-workers", type=int, default=4, help="Number of parallel workers for cropping")
    batch_parser.add_argument("--transcribe-workers", type=int, default=8, help="Number of concurrent workers for transcription")
    batch_parser.add_argument("--overwrite-crops", action="store_true", help="Re-crop pages even if PNGs exist")
    batch_parser.add_argument("--overwrite-cache", action="store_true", help="Re-transcribe columns even if cache TXTs exist")

    # Command: crop (Cropping only)
    crop_parser = subparsers.add_parser("crop", help="Batch crop PDF pages into column images")
    crop_parser.add_argument("--pdf", type=str, default="An_Uyghur_English_Dictionary_Complete.pdf", help="Source PDF file")
    crop_parser.add_argument("--start", type=int, default=26, help="Start PDF page (default: 26)")
    crop_parser.add_argument("--end", type=int, default=972, help="End PDF page (default: 972)")
    crop_parser.add_argument("--crops-dir", type=str, default="crops", help="Crops directory")
    crop_parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    crop_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crop images")

    # Command: transcribe (Transcription only)
    trans_parser = subparsers.add_parser("transcribe", help="Batch transcribe column crops concurrently")
    trans_parser.add_argument("--start", type=int, default=26, help="Start PDF page (default: 26)")
    trans_parser.add_argument("--end", type=int, default=972, help="End PDF page (default: 972)")
    trans_parser.add_argument("--crops-dir", type=str, default="crops", help="Crops directory")
    trans_parser.add_argument("--cache-dir", type=str, default="transcription_cache", help="Cache directory")
    trans_parser.add_argument("--workers", type=int, default=8, help="Number of concurrent API worker threads")
    trans_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached transcriptions")

    # Command: build (DB and JSONL build only)
    build_parser = subparsers.add_parser("build", help="Assemble SQLite DB and JSONL directly from transcription cache")
    build_parser.add_argument("--start", type=int, default=26, help="Start PDF page (default: 26)")
    build_parser.add_argument("--end", type=int, default=972, help="End PDF page (default: 972)")
    build_parser.add_argument("--cache-dir", type=str, default="transcription_cache", help="Cache directory")
    build_parser.add_argument("--output", type=str, default="uyghur_english_dictionary", help="Output prefix")

    # Command: process (Legacy staged run)
    proc_parser = subparsers.add_parser("process", help="Process a range of pages sequentially")
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

    if args.command == "batch":
        pipeline.batch_process_dictionary(
            pdf_path=args.pdf,
            start_page_pdf=args.start,
            end_page_pdf=args.end,
            crops_dir=args.crops_dir,
            cache_dir=args.cache_dir,
            output_prefix=args.output,
            crop_workers=args.crop_workers,
            transcribe_workers=args.transcribe_workers,
            overwrite_crops=args.overwrite_crops,
            overwrite_cache=args.overwrite_cache
        )

    elif args.command == "crop":
        print(f"Batch cropping PDF pages {args.start} to {args.end} into '{args.crops_dir}' with {args.workers} workers...")
        res = cropper.batch_crop_pages(
            pdf_path=args.pdf,
            start_page_pdf=args.start,
            end_page_pdf=args.end,
            output_dir=args.crops_dir,
            max_workers=args.workers,
            overwrite=args.overwrite,
            show_progress=True
        )
        print(f"[DONE] Cropped {len(res)} pages ({len(res) * 2} columns) into '{args.crops_dir}'.")

    elif args.command == "transcribe":
        tasks = []
        for p in range(args.start, args.end + 1):
            tasks.append({
                "page_pdf": p,
                "col": 1,
                "image_path": os.path.join(args.crops_dir, f"page_{p}_col1.png"),
                "cache_path": os.path.join(args.cache_dir, f"page_{p}_col1.txt")
            })
            tasks.append({
                "page_pdf": p,
                "col": 2,
                "image_path": os.path.join(args.crops_dir, f"page_{p}_col2.png"),
                "cache_path": os.path.join(args.cache_dir, f"page_{p}_col2.txt")
            })
        print(f"Batch transcribing {len(tasks)} columns with {args.workers} concurrent workers...")
        pipeline.batch_transcribe_columns(
            column_tasks=tasks,
            max_workers=args.workers,
            overwrite=args.overwrite,
            show_progress=True
        )
        print(f"[DONE] Batch transcription completed. Cached in '{args.cache_dir}'.")

    elif args.command == "build":
        print(f"Building dictionary from cache for pages {args.start} to {args.end}...")
        pipeline.build_from_cache(
            start_page_pdf=args.start,
            end_page_pdf=args.end,
            cache_dir=args.cache_dir,
            output_prefix=args.output
        )

    elif args.command == "process":
        print(f"Processing PDF pages {args.start} to {args.end}...")
        entries = pipeline.process_page_range(
            start_page_pdf=args.start,
            end_page_pdf=args.end,
            crops_dir=args.crops_dir,
            cache_dir=args.cache_dir,
            output_prefix=args.output
        )
        print(f"Processed {len(entries)} entries. Saved to {args.output}.jsonl and {args.output}.sqlite.")

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
