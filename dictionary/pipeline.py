"""
pipeline.py - Orchestrator for Uyghur-English Dictionary Digitization Pipeline
Supports staged execution, caching, column stitching, script conversion, and export.
"""

import os
import time
import re
import base64
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import fitz
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

import cropper
import parser
import converter
import db_builder

load_dotenv()

API_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-3.7-flash"
REASONING_EFFORT = "medium"

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it in .env or your environment.")
    return OpenAI(base_url=API_URL, api_key=api_key)

SYSTEM_PROMPT = """You are an expert lexicographical transcription engine for Henry G. Schwarz's Uyghur-English Dictionary.
The book uses a custom Latin alphabet with these 32 letters:
- Vowels: a, ä (a with umlaut), o, ö (o with umlaut), u, ü (u with umlaut), e (e without umlaut), i
- Consonants: b, p, t, j, ç (c with cedilla), x, d, r, z, ⱬ (z with stroke / Uyghur zh), s, ş (s with cedilla), g̃ (g with tilde / Uyghur gh), f, q, k, g, ñ (n with tilde / Uyghur ng), l, m, n, h, w, y

CRITICAL TRANSCRIPTION RULES:
1. Differentiate the bold leadword on the top left/right corner from the headwords. It has a bigger font size. The leadword is a header and not the main text. Therefore, don't include it in the output.
1. Format all headwords in **bold** (e.g. **abdal II**, **abistirakitlaş-**, **ⱬargon**, **g̃arañ-g̃uruñ**).
2. If the start of the main text is not a headword but (a continuation of) a definition from the previous column/page, prefix that line with [CONTINUATION].
3. Watch out for all tricky diacritics:
   - Cedillas on ç and ş: do not confuse with c or s.
   - Umlauts on ä, ö, ü: do not omit dots (e.g. şöhrät, mötiwär, kötirildi).
   - Tildes on ñ and g̃: do not omit tildes (e.g. maqaliniñ, g̃arañ-g̃uruñ).
   - Stroke/descender on ⱬ: do not confuse with z (e.g. ⱬargon, ⱬañ, ⱬornal).
   - Dotted i vs dotless ı: Uyghur only has dotted i.
4. Preserve numbered senses (1., 2.), sub-senses (a., b.), parts of speech (n., adj., vi., vt., etc.), domain labels (bot., rel., ag., geo., etc.), italicized Uyghur examples (e.g. *pulni apirip bärdim*), translations, idioms (marked with ♦), proverbs (marked with ¶), cross-references (Cf., Also, Also called), and etymology brackets ([...]) in the definition.
5. Dehyphenate line breaks: Automatically rejoin single words that are split across line wraps with a hyphen (e.g., "pres- / tige" → "prestige", "mendi- / cancy" → "mendicancy"). Do not remove true lexical/grammatical hyphens.

Transcribe all entries in this column image verbatim."""


def stitch_body_text(prev_body: str, continuation: str) -> str:
    """Stitch broken body text across columns or pages, dehyphenating wrapped words."""
    continuation = re.sub(r'^\s*\[CONTINUATION\]\s*', '', continuation).strip()
    prev_body = prev_body.strip()
    if not prev_body:
        return continuation
    if not continuation:
        return prev_body
    
    # If prev_body ends with a word-split hyphen e.g. 'aris-' and continuation starts with lowercase letter e.g. 'tocratic'
    m = re.search(r'([a-zA-Zäöüçşg̃ⱬñÄÖÜÇŞG̃Ñ]+)-\s*$', prev_body)
    if m and re.match(r'^[a-zäöüçşg̃ⱬñ]', continuation):
        cut_len = len(prev_body) - m.end() + 1
        return prev_body[:len(prev_body) - cut_len] + continuation
    else:
        return prev_body + " " + continuation


def transcribe_column_image(
    image_path: str,
    cache_path: Optional[str] = None,
    client: Optional[OpenAI] = None,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    overwrite_cache: bool = False
) -> str:
    """Send a single column image to the multimodal API with caching and retry logic."""
    if not overwrite_cache and cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached_text = f.read()
            if cached_text.strip():
                return cached_text

    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    if client is None:
        client = get_client()

    last_err = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                            {"type": "text", "text": "Transcribe all dictionary entries in this column image verbatim."}
                        ]
                    }
                ],
                reasoning_effort=REASONING_EFFORT,
            )

            content = response.choices[0].message.content.strip()

            if content.startswith('```text') or content.startswith('```markdown') or content.startswith('```'):
                content = re.sub(r'^```[a-zA-Z]*\n', '', content)
                content = re.sub(r'\n```$', '', content).strip()

            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            return content
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                sleep_time = (retry_delay * (2 ** attempt)) + random.uniform(0.5, 1.5)
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"Failed to transcribe {image_path} after {max_retries} attempts: {last_err}") from last_err


def batch_transcribe_columns(
    column_tasks: List[Dict[str, Any]],
    max_workers: int = 8,
    overwrite: bool = False,
    show_progress: bool = True
) -> Dict[str, str]:
    """
    Concurrent transcription for a list of column tasks.
    Each task dict:
      - "image_path": path to image PNG
      - "cache_path": path to cache TXT
      - "page_pdf": int
      - "col": int
    """
    results: Dict[str, str] = {}
    tasks_to_run = []
    cached_count = 0

    for task in column_tasks:
        img_p = task["image_path"]
        c_p = task.get("cache_path")
        if not overwrite and c_p and os.path.exists(c_p):
            with open(c_p, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    results[img_p] = content
                    cached_count += 1
                    continue
        tasks_to_run.append(task)

    if show_progress:
        print(f"Batch transcription: {len(column_tasks)} total columns | {cached_count} cached | {len(tasks_to_run)} to transcribe.")

    if not tasks_to_run:
        return results

    client = get_client()

    def _worker(task: Dict[str, Any]) -> Tuple[str, str]:
        text = transcribe_column_image(
            image_path=task["image_path"],
            cache_path=task.get("cache_path"),
            client=client,
            overwrite_cache=overwrite
        )
        return task["image_path"], text

    if max_workers <= 1:
        iterable = (_worker(t) for t in tasks_to_run)
        if show_progress:
            iterable = tqdm(iterable, total=len(tasks_to_run), desc="Transcribing columns")
        for img_p, text in iterable:
            results[img_p] = text
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(_worker, t): t for t in tasks_to_run}
            if show_progress:
                pbar = tqdm(total=len(tasks_to_run), desc=f"Transcribing ({max_workers} threads)")
                for future in as_completed(future_to_task):
                    img_p, text = future.result()
                    results[img_p] = text
                    pbar.update(1)
                pbar.close()
            else:
                for future in as_completed(future_to_task):
                    img_p, text = future.result()
                    results[img_p] = text

    return results


def build_entries_from_transcriptions(
    page_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Stitch and parse dictionary entries from a sequential list of page transcriptions.
    page_data: list of dicts {"page_pdf": int, "c1_text": str, "c2_text": str} sorted by page_pdf.
    """
    all_entries: List[Dict[str, Any]] = []

    for item in page_data:
        p_num = item["page_pdf"]
        page_book = p_num - 25
        c1_text = item["c1_text"]
        c2_text = item["c2_text"]

        # Parse Column 1
        cont1, entries1, _ = parser.parse_column_transcription(
            c1_text, page_book=page_book, page_pdf=p_num, column=1
        )

        # Stitch continuation from previous page
        if cont1 and all_entries:
            last_entry = all_entries[-1]
            last_entry["raw_body"] = stitch_body_text(last_entry["raw_body"], cont1)
            reparsed = parser.parse_raw_entry_text(
                last_entry["headword"]["schwarz"],
                last_entry["homograph_number"],
                [last_entry["raw_body"]],
                last_entry["page_book"],
                last_entry["page_pdf"],
                last_entry["column"]
            )
            all_entries[-1] = reparsed

        # Parse Column 2
        cont2, entries2, _ = parser.parse_column_transcription(
            c2_text, page_book=page_book, page_pdf=p_num, column=2
        )

        target_list = entries1 if entries1 else all_entries
        if cont2 and target_list:
            last_entry = target_list[-1]
            last_entry["raw_body"] = stitch_body_text(last_entry["raw_body"], cont2)
            reparsed = parser.parse_raw_entry_text(
                last_entry["headword"]["schwarz"],
                last_entry["homograph_number"],
                [last_entry["raw_body"]],
                last_entry["page_book"],
                last_entry["page_pdf"],
                last_entry["column"]
            )
            target_list[-1] = reparsed

        all_entries.extend(entries1)
        all_entries.extend(entries2)

    return all_entries


def build_from_cache(
    start_page_pdf: int = 26,
    end_page_pdf: int = 972,
    cache_dir: str = "transcription_cache",
    output_prefix: str = "uyghur_english_dictionary"
) -> List[Dict[str, Any]]:
    """Build SQLite DB and JSONL directly from cached column transcriptions."""
    page_data: List[Dict[str, Any]] = []
    missing: List[str] = []

    for p_num in range(start_page_pdf, end_page_pdf + 1):
        c1_cache = os.path.join(cache_dir, f"page_{p_num}_col1.txt")
        c2_cache = os.path.join(cache_dir, f"page_{p_num}_col2.txt")

        c1_text = ""
        c2_text = ""
        if os.path.exists(c1_cache):
            with open(c1_cache, "r", encoding="utf-8") as f:
                c1_text = f.read()
        else:
            missing.append(c1_cache)

        if os.path.exists(c2_cache):
            with open(c2_cache, "r", encoding="utf-8") as f:
                c2_text = f.read()
        else:
            missing.append(c2_cache)

        page_data.append({
            "page_pdf": p_num,
            "c1_text": c1_text,
            "c2_text": c2_text
        })

    if missing:
        print(f"[WARN] {len(missing)} column cache files not found. Sample missing: {missing[:3]}")

    all_entries = build_entries_from_transcriptions(page_data)

    # Save outputs
    jsonl_out = f"{output_prefix}.jsonl"
    db_out = f"{output_prefix}.sqlite"
    db_builder.save_dictionary_to_db_and_jsonl(all_entries, db_path=db_out, jsonl_path=jsonl_out)

    print(f"\n[DONE] Successfully built {len(all_entries)} entries across pages {start_page_pdf}-{end_page_pdf}!")
    print(f"  JSONL output: {jsonl_out}")
    print(f"  SQLite DB output: {db_out}")
    return all_entries


def batch_process_dictionary(
    pdf_path: str = "An_Uyghur_English_Dictionary_Complete.pdf",
    start_page_pdf: int = 26,
    end_page_pdf: int = 972,
    crops_dir: str = "crops",
    cache_dir: str = "transcription_cache",
    output_prefix: str = "uyghur_english_dictionary",
    crop_workers: int = 4,
    transcribe_workers: int = 8,
    overwrite_crops: bool = False,
    overwrite_cache: bool = False
) -> List[Dict[str, Any]]:
    """End-to-end parallel batch pipeline: Cropping -> Transcription -> Stitching & DB Build."""
    print("=" * 65)
    print(f"UYGHUR-ENGLISH DICTIONARY BATCH DIGITALIZATION (Pages {start_page_pdf} to {end_page_pdf})")
    print("=" * 65)

    # Stage 1: Parallel Cropping
    print(f"\n[STAGE 1/3] Cropping pages {start_page_pdf} to {end_page_pdf}...")
    crop_results = cropper.batch_crop_pages(
        pdf_path=pdf_path,
        start_page_pdf=start_page_pdf,
        end_page_pdf=end_page_pdf,
        output_dir=crops_dir,
        max_workers=crop_workers,
        overwrite=overwrite_crops,
        show_progress=True
    )

    # Stage 2: Concurrent Transcription
    print(f"\n[STAGE 2/3] Transcribing {len(crop_results) * 2} columns with {transcribe_workers} workers...")
    tasks = []
    for cr in crop_results:
        p_num = cr["page_pdf"]
        tasks.append({
            "page_pdf": p_num,
            "col": 1,
            "image_path": cr["col1_path"],
            "cache_path": os.path.join(cache_dir, f"page_{p_num}_col1.txt")
        })
        tasks.append({
            "page_pdf": p_num,
            "col": 2,
            "image_path": cr["col2_path"],
            "cache_path": os.path.join(cache_dir, f"page_{p_num}_col2.txt")
        })

    batch_transcribe_columns(
        column_tasks=tasks,
        max_workers=transcribe_workers,
        overwrite=overwrite_cache,
        show_progress=True
    )

    # Stage 3: Stitching, Parsing, and Database Construction
    print(f"\n[STAGE 3/3] Parsing transcriptions, stitching continuations, and building DB...")
    all_entries = build_from_cache(
        start_page_pdf=start_page_pdf,
        end_page_pdf=end_page_pdf,
        cache_dir=cache_dir,
        output_prefix=output_prefix
    )

    return all_entries


def process_page_range(
    pdf_path: str = "An_Uyghur_English_Dictionary_Complete.pdf",
    start_page_pdf: int = 56,
    end_page_pdf: int = 56,
    crops_dir: str = "crops",
    cache_dir: str = "transcription_cache",
    output_prefix: str = "staged_test"
) -> List[Dict[str, Any]]:
    """Process a range of PDF pages, stitch entries, and output JSONL + SQLite DB."""
    doc = fitz.open(pdf_path)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    page_data: List[Dict[str, Any]] = []

    for p_num in range(start_page_pdf, end_page_pdf + 1):
        print(f"--> Processing PDF Page {p_num} (Book Page {p_num - 25})...")
        page = doc[p_num - 1]

        crop_info = cropper.crop_page_columns(page, crops_dir, p_num)

        # Column 1
        c1_cache = os.path.join(cache_dir, f"page_{p_num}_col1.txt")
        t0 = time.time()
        c1_text = transcribe_column_image(crop_info["col1_path"], c1_cache)
        t1 = time.time()
        print(f"    Col 1 transcribed in {t1 - t0:.2f}s")

        # Column 2
        c2_cache = os.path.join(cache_dir, f"page_{p_num}_col2.txt")
        t0 = time.time()
        c2_text = transcribe_column_image(crop_info["col2_path"], c2_cache)
        t1 = time.time()
        print(f"    Col 2 transcribed in {t1 - t0:.2f}s")

        page_data.append({
            "page_pdf": p_num,
            "c1_text": c1_text,
            "c2_text": c2_text
        })

    all_entries = build_entries_from_transcriptions(page_data)

    # Save outputs
    jsonl_out = f"{output_prefix}.jsonl"
    db_out = f"{output_prefix}.sqlite"
    db_builder.save_dictionary_to_db_and_jsonl(all_entries, db_path=db_out, jsonl_path=jsonl_out)

    print(f"\n[DONE] Successfully processed {len(all_entries)} entries across pages {start_page_pdf}-{end_page_pdf}!")
    print(f"  JSONL output: {jsonl_out}")
    print(f"  SQLite DB output: {db_out}")
    return all_entries
