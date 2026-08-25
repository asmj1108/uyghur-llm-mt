"""
pipeline.py - Orchestrator for Uyghur-English Dictionary Digitization Pipeline
Supports staged execution, caching, column stitching, script conversion, and export.
"""

import os
import time
import re
import base64
from typing import List, Dict, Any, Optional

import fitz
from openai import OpenAI

import cropper
import parser
import converter
import db_builder

API_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "google/gemini-3.7-flash"
REASONING_EFFORT = "medium"

def get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    return OpenAI(base_url=API_URL, api_key=api_key)

SYSTEM_PROMPT = """You are an expert lexicographical transcription engine for Henry G. Schwarz's Uyghur-English Dictionary.
The book uses a custom Latin alphabet with these 32 letters:
- Vowels: a, ä (a with umlaut), o, ö (o with umlaut), u, ü (u with umlaut), e (e without umlaut), i
- Consonants: b, p, t, j, ç (c with cedilla), x, d, r, z, ⱬ (z with stroke / Uyghur zh), s, ş (s with cedilla), g̃ (g with tilde / Uyghur gh), f, q, k, g, ñ (n with tilde / Uyghur ng), l, m, n, h, w, y

CRITICAL TRANSCRIPTION RULES:
1. Format all main headwords in **bold** (e.g. **abdal**, **abistirakitlaş-**, **ⱬargon**, **g̃arañ-g̃uruñ**).
2. If the top line is a continuation of a definition from the previous column/page, prefix that line with [CONTINUATION].
3. Watch out for all tricky diacritics:
   - Cedillas on ç and ş: do not confuse with c or s.
   - Umlauts on ä, ö, ü: do not omit dots (e.g. şöhrät, mötiwär, kötirildi).
   - Tildes on ñ and g̃: do not omit tildes (e.g. maqaliniñ, g̃arañ-g̃uruñ).
   - Stroke/descender on ⱬ: do not confuse with z (e.g. ⱬargon, ⱬañ, ⱬornal).
   - Dotted i vs dotless ı: Uyghur only has dotted i.
4. Preserve numbered senses (1., 2.), sub-senses (a., b.), parts of speech (n., adj., vi., vt., etc.), domain labels (bot., rel., ag., geo., etc.), italicized Uyghur examples (e.g. *pulni apirip bärdim*), translations, idioms (marked with ♦), proverbs (marked with ¶), cross-references (Cf., Also, Also called), and etymology brackets ([...]).
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


def transcribe_column_image(image_path: str, cache_path: Optional[str] = None) -> str:
    """Send a single column image to the multimodal API with caching."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            return f.read()

    with open(image_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    client = get_client()
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


def process_page_range(
    pdf_path: str = "An_Uyghur_English_Dictionary_Complete.pdf",
    start_page_pdf: int = 56,
    end_page_pdf: int = 56,
    crops_dir: str = "crops",
    cache_dir: str = "transcription_cache",
    output_prefix: str = "staged_test"
) -> List[Dict[str, Any]]:
    """Process a range of PDF pages, stitch entries, and output JSON + SQLite DB."""
    doc = fitz.open(pdf_path)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    all_entries: List[Dict[str, Any]] = []
    pending_continuation: Optional[str] = None

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

        # Parse Column 1
        cont1, entries1, _ = parser.parse_column_transcription(
            c1_text, page_book=p_num - 25, page_pdf=p_num, column=1
        )

        # Stitch continuation or resolve orphan headword from previous page
        if cont1:
            if all_entries:
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
        elif all_entries and not all_entries[-1]["raw_body"].strip():
            # If the last entry had an empty body (orphan headword at bottom of previous page),
            # the first entry in this column is the orphan's body with a hallucinated headword.
            if entries1:
                orphan_entry = all_entries[-1]
                first_entry = entries1[0]
                orphan_entry["raw_body"] = first_entry["raw_body"]
                reparsed = parser.parse_raw_entry_text(
                    orphan_entry["headword"]["schwarz"],
                    orphan_entry["homograph_number"],
                    [orphan_entry["raw_body"]],
                    orphan_entry["page_book"],
                    orphan_entry["page_pdf"],
                    orphan_entry["column"]
                )
                all_entries[-1] = reparsed
                entries1 = entries1[1:]

        # Parse Column 2
        cont2, entries2, _ = parser.parse_column_transcription(
            c2_text, page_book=p_num - 25, page_pdf=p_num, column=2
        )

        target_list = entries1 if entries1 else all_entries
        if cont2:
            if target_list:
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
        else:
            if target_list and not target_list[-1]["raw_body"].strip():
                if entries2:
                    orphan_entry = target_list[-1]
                    first_entry = entries2[0]
                    orphan_entry["raw_body"] = first_entry["raw_body"]
                    reparsed = parser.parse_raw_entry_text(
                        orphan_entry["headword"]["schwarz"],
                        orphan_entry["homograph_number"],
                        [orphan_entry["raw_body"]],
                        orphan_entry["page_book"],
                        orphan_entry["page_pdf"],
                        orphan_entry["column"]
                    )
                    target_list[-1] = reparsed
                    entries2 = entries2[1:]

        all_entries.extend(entries1)
        all_entries.extend(entries2)

    # Save outputs
    json_out = f"{output_prefix}.json"
    db_out = f"{output_prefix}.sqlite"
    db_builder.save_dictionary_to_db_and_json(all_entries, db_path=db_out, json_path=json_out)
    
    print(f"\n[DONE] Successfully processed {len(all_entries)} entries across pages {start_page_pdf}-{end_page_pdf}!")
    print(f"  JSON output: {json_out}")
    print(f"  SQLite DB output: {db_out}")
    return all_entries
