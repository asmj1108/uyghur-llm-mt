"""
parser.py - Comprehensive lexicographical parser for Henry G. Schwarz's Uyghur-English Dictionary
Parses definitions, examples, sayings, idioms, cross-references, etymologies, and correspondences.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import converter

# All 22 Subject / Restrictive domain labels from the book (page 25)
DOMAIN_TAGS = {
    'ag.': 'agriculture',
    'ast.': 'astronomy',
    'bot.': 'botany',
    'chem.': 'chemistry',
    'clo.': 'clothing',
    'cul.': 'culinary',
    'fig.': 'figurative',
    'geo.': 'geography',
    'geol.': 'geology',
    'gr.': 'grammar; linguistics',
    'his.': 'historical; history',
    'lit.': 'literature and arts',
    'math.': 'mathematics',
    'med.': 'medicine',
    'mil.': 'military',
    'mus.': 'music',
    'obs.': 'obsolete',
    'pe.': 'physical education; sports',
    'ph.': 'philosophy',
    'phy.': 'physics',
    'phys.': 'physiology',
    'poe.': 'poetic',
    'rel.': 'religion',
    'zoo.': 'zoology'
}

# Parts of speech from the book
POS_TAGS = {
    'adj.': 'adjective',
    'adv.': 'adverb',
    'conj.': 'conjunction',
    'int.': 'interjection',
    'm.': 'measure word',
    'num.': 'numeral',
    'n.': 'noun',
    'onom.': 'onomatepia',
    'pp.': 'postposition',
    'prep.': 'preposition',
    'pro.': 'pronoun',
    'vi.': 'intransitive verb',
    'vt.': 'transitive verb'
}

# Grammatical voice tags
VOICE_TAGS = {
    'caus.': 'causative voice',
    'mut.': 'mutual voice',
    'pass.': 'passive voice',
    'retro.': 'retroactive voice'
}

# 14 Turkic/Mongolic systems and loanword donor languages
LANG_NAMES = {
    'Ç': 'Chagatay',
    'Ch': 'Chinese',
    'Chin': 'Chinese',
    'E': 'English',
    'F': 'French',
    'G': 'German',
    'K': 'Kirghiz',
    'KK': 'Karakalpak',
    'M': 'Mahmud al-Kaşğari',
    'Mo': 'Mongolian',
    'O': 'Osmanli Turkish',
    'OT': 'Orkhon Turkic',
    'Q': 'Kazakh',
    'R': 'Russian',
    'Ta': 'Tatar',
    'TT': 'Turfan texts',
    'Tu': 'Turkmen',
    'Tv': 'Tuvan',
    'U': 'Uzbek',
    'Y': 'Yakut',
    'Ar': 'Arabic',
    'Pers': 'Persian'
}




def clean_markdown_formatting(text: str) -> str:
    """Strip markdown bold/italic asterisks, underscores, and backticks from text."""
    if not text:
        return ""
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = text.replace('**', '').replace('*', '').replace('`', '')
    return text.strip()


def normalize_punctuation(text: str) -> str:
    """Normalize multiple periods, extra spaces before punctuation, etc."""
    if not text:
        return ""
    # Replace multiple periods (with optional whitespace) with a single period
    text = re.sub(r'\.(\s*\.)+', '.', text)
    # Remove space before punctuation marks: .,;:!?
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_headword_line(line: str) -> Tuple[str, Optional[str]]:
    """Extract headword and optional homograph Roman numeral (I, II, III, etc.)."""
    cleaned = clean_markdown_formatting(line)
    m = re.match(r'^(.+?)\s+([IVXLCDM]+)$', cleaned)
    if m:
        hw = converter.normalize_schwarz_text(m.group(1).strip())
        return hw, m.group(2).strip()
    return converter.normalize_schwarz_text(cleaned), None


def reembed_headword(text: str, headword: str) -> str:
    """
    Replace '--', '—', or '–' placeholders in Uyghur examples, sayings, and idioms
    with the full headword or inflected stem.
    """
    if '--' not in text and '—' not in text and '–' not in text:
        return text

    is_verb = headword.endswith('-')
    stem = headword[:-1] if is_verb else headword

    def replace_suffix(match):
        prefix = match.group(1)
        suffix = match.group(2)
        
        if is_verb:
            # Verb inflections
            if suffix.startswith('p') and not stem.endswith(('a', 'ä', 'e', 'i', 'o', 'ö', 'u', 'ü')):
                vowel = 'i' if any(v in stem for v in ('i', 'e', 'ä', 'ö', 'ü')) else 'u'
                return f"{prefix}{stem}{vowel}p"
            elif suffix.startswith('mi-'):
                clean_suf = suffix.replace('-', '')
                return f"{prefix}{stem}{clean_suf}"
            return f"{prefix}{stem}{suffix}"
        else:
            # Noun / adjective inflections
            if stem.endswith('t') and suffix.startswith('tin'):
                return f"{prefix}attin"
            elif stem.endswith('t') and suffix.startswith('qa'):
                return f"{prefix}atqa"
            return f"{prefix}{stem}{suffix}"

    # Replace placeholders with attached suffixes e.g. *--lär, --i, --ni, (--qa)
    # Preceded by start of string or any non-alphanumeric character
    res = re.sub(r'([^\w\-]|^)(?:--|—|–)([a-zA-Zäöüçşg̃ⱬñÄÖÜÇŞG̃Ñ\-]+)', replace_suffix, text)
    
    # Standalone placeholder
    if is_verb:
        res = re.sub(r'([^\w\-]|^)(?:--|—|–)(?![\w\-])', f"\\1{stem}-", res)
    else:
        res = re.sub(r'([^\w\-]|^)(?:--|—|–)(?![\w\-])', f"\\1{stem}", res)

    return res


def parse_bracketed_info(text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate Etymologies (loanword chains with '<') and Correspondences (cognates without '<').
    Formatting (asterisks) is cleaned from word field.
    """
    brackets = re.findall(r'\[([^\]]+)(?:\]|$)', text)
    etymologies = []
    correspondences = []

    for b in brackets:
        b = b.strip()
        if not b:
            continue

        if '<' in b:
            # Loanword etymology chain, e.g. "<R. abonement <F. abonne-"
            steps = re.split(r'\s*<\s*', b)
            chain = []
            for step in steps:
                step = step.strip()
                if not step:
                    continue
                m = re.match(r'^([A-ZÇ][a-zA-Z]*)\.\s*(.*)', step)
                if m:
                    lang_code = m.group(1)
                    raw_word = m.group(2).strip()
                    word = clean_markdown_formatting(raw_word)
                    chain.append({
                        "lang_code": lang_code,
                        "language": LANG_NAMES.get(lang_code, lang_code),
                        "word": word,
                    })
                else:
                    chain.append({"raw": clean_markdown_formatting(step)})
            etymologies.append({"raw": b, "chain": chain})
        else:
            # Correspondence cognates across Turkic/Mongolic systems, e.g. "Ç. at/K. at/U. ot"
            items = re.split(r'\s*/\s*', b)
            for item in items:
                item = item.strip()
                if not item:
                    continue
                m = re.match(r'^([A-ZÇ][a-zA-Z]*)\.\s*([^(\[]+)(.*)', item)
                if m and m.group(1) in LANG_NAMES:
                    lang_code = m.group(1)
                    raw_word_part = m.group(2).strip()
                    raw_note_part = m.group(3).strip()

                    word_part = clean_markdown_formatting(raw_word_part)
                    note_part = clean_markdown_formatting(raw_note_part)

                    words = [w.strip() for w in word_part.split(';') if w.strip()]
                    for w in words:
                        correspondences.append({
                            "lang_code": lang_code,
                            "language": LANG_NAMES.get(lang_code, lang_code),
                            "word": w,
                            "note": note_part if note_part else None
                        })
                else:
                    m2 = re.match(r'^([A-ZÇ][a-zA-Z]*)\.\s*(.*)', item)
                    if m2 and m2.group(1) in LANG_NAMES:
                        lang_code = m2.group(1)
                        word = clean_markdown_formatting(m2.group(2).strip())
                        correspondences.append({
                            "lang_code": lang_code,
                            "language": LANG_NAMES.get(lang_code, lang_code),
                            "word": word,
                            "note": None
                        })

    return etymologies, correspondences


def parse_cross_references(text: str) -> List[Dict[str, Any]]:
    """
    Extract cross-references (Also, Also called, Cf.) and convert targets to UEY + ULY.
    """
    cross_refs = []
    for m in re.finditer(r'\b(Also called|Also|Cf\.)\s+([^.;\[]+)', text):
        ref_type = m.group(1).strip()
        targets_str = clean_markdown_formatting(m.group(2).strip())
        
        targets = [t.strip() for t in targets_str.split(',') if t.strip()]
        for t in targets:
            norm_t = converter.normalize_schwarz_text(t)
            cross_refs.append({
                "type": ref_type,
                "target_schwarz": norm_t,
                "target_uey": converter.custom_to_arabic(norm_t),
                "target_uly": converter.custom_to_uly(norm_t)
            })
    return cross_refs


def parse_sayings(text: str, headword: str) -> Tuple[List[Dict[str, Any]], str]:
    """Extract proverbs/sayings marked with ¶ and remove from remaining text."""
    sayings = []
    # Match ONLY ¶ (Pilcrow), not character 5 or other numbers!
    saying_matches = re.finditer(r'¶\s*([^/♦\[]+)', text)
    for m in saying_matches:
        full_saying = m.group(1).strip()
        
        explanation = None
        m_exp = re.search(r'\(([^)]+)\)\s*$', full_saying)
        if m_exp:
            explanation = m_exp.group(1).strip()
            full_saying = full_saying[:m_exp.start()].strip()

        embedded_saying = reembed_headword(full_saying, headword)
        
        italic_matches = list(re.finditer(r'(?<!\*)\*([^*]+)\*(?!\*)', embedded_saying))
        if italic_matches:
            last_italic_end = italic_matches[-1].end()
            uyghur_raw = clean_markdown_formatting(embedded_saying[:last_italic_end].strip())
            english = normalize_punctuation(clean_markdown_formatting(embedded_saying[last_italic_end:].strip()))
        else:
            uyghur_raw = clean_markdown_formatting(embedded_saying)
            english = ""

        sayings.append({
            "uyghur_schwarz": uyghur_raw,
            "uyghur_uey": converter.custom_to_arabic(uyghur_raw),
            "uyghur_uly": converter.custom_to_uly(uyghur_raw),
            "english": english,
            "explanation": explanation
        })

    cleaned_text = re.sub(r'¶\s*[^/♦\[]+', '', text)
    return sayings, cleaned_text


def parse_special_meanings(text: str, headword: str) -> Tuple[List[Dict[str, Any]], str]:
    """Extract idioms/special terms marked with ♦ and remove from remaining text."""
    special_meanings = []
    sm_matches = re.finditer(r'♦\s*([^/¶\[]+)', text)
    for m in sm_matches:
        full_sm = m.group(1).strip()
        # Split on colon if followed by a sub-expression/example
        sub_items = re.split(r'\s*:\s*(?=(?:--|—|–|\*))', full_sm)

        prefix_desc = ""
        for idx, item in enumerate(sub_items):
            item = item.strip()
            if not item:
                continue

            # If the first item has no Uyghur marker (* or --) and there are subsequent items,
            # treat it as a descriptive prefix (e.g. 'to blemish:')
            has_marker = bool(re.search(r'\*[^*]+\*|--|—|–', item))
            if idx == 0 and len(sub_items) > 1 and not has_marker:
                prefix_desc = clean_markdown_formatting(item)
                continue

            embedded_sm = reembed_headword(item, headword)

            italic_matches = list(re.finditer(r'(?<!\*)\*([^*]+)\*(?!\*)', embedded_sm))
            if italic_matches:
                last_italic_end = italic_matches[-1].end()
                uyghur_raw = clean_markdown_formatting(embedded_sm[:last_italic_end].strip())
                english = normalize_punctuation(clean_markdown_formatting(embedded_sm[last_italic_end:].strip()))
            else:
                uyghur_raw = clean_markdown_formatting(embedded_sm)
                english = ""

            if prefix_desc:
                if english:
                    english = f"{prefix_desc}: {english}"
                else:
                    english = prefix_desc
                prefix_desc = ""

            if not uyghur_raw:
                continue

            special_meanings.append({
                "uyghur_schwarz": uyghur_raw,
                "uyghur_uey": converter.custom_to_arabic(uyghur_raw),
                "uyghur_uly": converter.custom_to_uly(uyghur_raw),
                "english": english
            })

    cleaned_text = re.sub(r'♦\s*[^/¶\[]+', '', text)
    return special_meanings, cleaned_text


def parse_examples(text: str, headword: str) -> List[Dict[str, Any]]:
    """
    Extract example phrases separated by / and re-embed headword.
    Separates Uyghur phrase from English definition and domain tags.
    """
    examples = []
    parts = re.split(r'\s*/\s*', text)
    for p in parts:
        p = p.strip()
        if not p or p.startswith('Cf.') or p.startswith('Also'):
            continue
        
        # 1. Re-embed headword into example string
        embedded = reembed_headword(p, headword)
        
        # 2. Extract Uyghur vs English chunk (Uyghur is single-asterisk italicized, English is the remainder)
        italic_matches = list(re.finditer(r'(?<!\*)\*([^*]+)\*(?!\*)', embedded))
        if italic_matches:
            last_italic_end = italic_matches[-1].end()
            uyghur_chunk = embedded[:last_italic_end].strip()
            english_chunk = embedded[last_italic_end:].strip()
        else:
            uyghur_chunk = embedded
            english_chunk = ""

        # 3. Extract domain tag from English or Uyghur boundary
        domain = None
        for d_tag in DOMAIN_TAGS:
            if english_chunk.startswith(d_tag):
                domain = d_tag
                english_chunk = english_chunk[len(d_tag):].strip()
                break
            elif f" {d_tag} " in english_chunk:
                domain = d_tag
                english_chunk = english_chunk.replace(f" {d_tag} ", " ").strip()
                break

        uyghur_raw = clean_markdown_formatting(uyghur_chunk)
        english_raw = normalize_punctuation(clean_markdown_formatting(english_chunk))

        if not uyghur_raw and english_raw:
            uyghur_raw = english_raw
            english_raw = ""

        examples.append({
            "uyghur_schwarz": uyghur_raw,
            "uyghur_uey": converter.custom_to_arabic(uyghur_raw),
            "uyghur_uly": converter.custom_to_uly(uyghur_raw),
            "domain": domain,
            "english": english_raw
        })
    return examples


def convert_definition_references(definition: str) -> str:
    """
    Convert cross-headword references in definitions to UEY:
    1. '= <headword> [number]' -> '= <headword_uey> [number]'
    2. 'abstract n. of <stem>' -> 'abstract n. of <stem_uey>'
    3. 'vi. caus. of <stem>' -> 'vi. caus. of <stem_uey>'
    4. 'Alternate name of <stem>' -> 'Alternate name of <stem_uey>'
    """
    if not definition:
        return ""

    # 1. Match "= <headword> [number]" with or without asterisks
    def replace_equals(m):
        prefix = m.group(1) # e.g. '= '
        hw = m.group(2).replace('*', '').strip()
        num = (m.group(3) or '').replace('*', '')
        punct = m.group(4) or ''
        uey = converter.custom_to_arabic(hw)
        return f"{prefix}{uey}{num}{punct}"

    pattern_equals = re.compile(
        r'(=\s*)(?:\*{1,2})?([a-zA-Z\u00C0-\u024F\u0300-\u036F\u2C6C\-\']+)(?:\*{1,2})?(\s+[IVXLCDM\d]+(?:\s+\d+)?)?(?:\*{1,2})?([.,;]?)',
        re.IGNORECASE
    )
    res = pattern_equals.sub(replace_equals, definition)

    # 2. Match grammatical 'of <stem>' derivation references
    pattern_of = re.compile(
        r"""(?P<prefix>
            (?:(?:abstract|collective|verbal|diminutive|dim\.?)\s+)?
            (?:adj|adv|conj|int|m|num|n|onom|pp|prep|pro|vi|vt|caus|mut|pass|retro|pl|plural|variant|var|syn|comp|superl)\.\s*
            (?:(?:caus|mut|pass|retro)\.\s*)?
            |
            (?:Alternate|alternate)\s+name\s*
        )
        of\s+
        (?:\*{1,2})?(?P<stem>[a-zA-Z\u00C0-\u024F\u0300-\u036F\u2C6C\-\']+)(?P<num>\s+[IVXLCDM\d]+)?(?:\*{1,2})?
        """,
        re.VERBOSE | re.IGNORECASE
    )

    def replace_of(m):
        prefix = m.group("prefix").strip()
        stem = m.group("stem").replace("*", "").strip()
        num = m.group("num") or ""
        uey = converter.custom_to_arabic(stem)
        return f"{prefix} of {uey}{num}"

    res = pattern_of.sub(replace_of, res)
    return res


def is_example_section(ex_part: str) -> bool:
    """Check if the text following a colon is truly an example section."""
    ex = ex_part.strip()
    if not ex:
        return False
    if any(m in ex for m in ('--', '—', '–')):
        return True
    if '*' in ex:
        return True
    if '/' in ex:
        return True
    if re.match(r'^[\d\s,.\-—–()]+$', ex):
        return False
    if any(w in ex.lower() for w in ('to ', 'a ', 'an ', 'the ')):
        return True
    return False


def parse_single_sense(sense_num: int, text: str, headword: str) -> Dict[str, Any]:
    """Parse a single sense into explanation, examples, sayings, special meanings, cross-refs, and etymologies."""
    # 1. Extract bracketed info (etymology vs correspondence)
    etymologies, correspondences = parse_bracketed_info(text)
    text_nobrackets = re.sub(r'\[.*?\]', '', text).strip()
    text_nobrackets = re.sub(r'\[.*$', '', text_nobrackets).strip()
    text_nobrackets = normalize_punctuation(text_nobrackets)

    # 2. Extract Cross-references
    cross_refs = parse_cross_references(text_nobrackets)
    text_nocross = re.sub(r'\b(Also called|Also|Cf\.)\s+[^.;]+[.;]?', '', text_nobrackets).strip()
    text_nocross = normalize_punctuation(text_nocross)

    # 3. Extract Sayings (¶) and Special Meanings (♦)
    sayings, text_nosayings = parse_sayings(text_nocross, headword)
    special_meanings, text_nosm = parse_special_meanings(text_nosayings, headword)
    text_nosm = normalize_punctuation(text_nosm)

    # 4. Extract POS, Domain, and Voice from leading tags
    pos = None
    voice = None
    domain = None
    
    # Check for compound grammatical prefix like "abstract n.", "collective n.", "verbal n.", "dim. n."
    m_compound = re.match(r'^(abstract|collective|verbal|diminutive|dim\.)\s+([a-z]+\.)\s*(.*)', text_nosm, re.IGNORECASE | re.DOTALL)
    if m_compound:
        tag = m_compound.group(2).lower()
        if tag in POS_TAGS:
            pos = tag
        definition_and_examples = text_nosm
    else:
        # Check standard leading tags e.g. "geo. n. ...", "bot. n. ...", "vt. ...", "adj. ..."
        m_tag = re.match(r'^(([a-z]+\.\s*){1,4})\s*(.*)', text_nosm, re.DOTALL)
        if m_tag:
            potential_tags = [t.strip() + '.' for t in m_tag.group(1).split('.') if t.strip()]
            for tag in potential_tags:
                if tag in POS_TAGS and not pos:
                    pos = tag
                elif tag in VOICE_TAGS and not voice:
                    voice = tag
                elif tag in DOMAIN_TAGS and not domain:
                    domain = tag
            
            rest = m_tag.group(3).strip()
            # If the rest starts with 'of ', preserve leading grammatical tag in definition
            if rest.startswith('of '):
                definition_and_examples = text_nosm
            else:
                definition_and_examples = rest
        else:
            definition_and_examples = text_nosm

    # 5. Separate English explanation from Examples (separated by :)
    definition_text = definition_and_examples
    examples = []
    
    if ':' in definition_and_examples:
        def_part, ex_part = definition_and_examples.split(':', 1)
        if is_example_section(ex_part):
            definition_text = def_part.strip()
            examples = parse_examples(ex_part.strip(), headword)
        else:
            definition_text = definition_and_examples
    else:
        definition_text = definition_and_examples

    # Convert references and clean markdown in definition
    definition_text = convert_definition_references(definition_text)
    definition_text = clean_markdown_formatting(definition_text)
    definition_text = normalize_punctuation(definition_text)

    return {
        "sense_number": sense_num,
        "pos": pos,
        "voice": voice,
        "domain": domain,
        "definition": definition_text,
        "examples": examples,
        "sayings": sayings,
        "special_meanings": special_meanings,
        "cross_references": cross_refs,
        "etymologies": etymologies,
        "correspondences": correspondences,
        "raw_text": text
    }


def split_into_numbered_senses(body_text: str) -> List[Tuple[int, str]]:
    """
    Split entry body text into numbered senses sequentially: 1., 2., 3., ...
    Ensures arbitrary years like '1983.' or homograph numbers in cross-refs are not misidentified.
    """
    body_text = body_text.strip()
    
    # Check if entry begins with '1. ' or '<POS/Domain> 1. '
    m_start = re.match(r'^(?:(?:[a-z]+\.\s*){0,3})1\.\s+(.*)', body_text, re.DOTALL)
    if not m_start:
        return [(1, body_text)]

    senses = []
    current_num = 1
    remaining = body_text
    
    # Remove leading POS tags before "1. " if any and attach to sense 1
    leading_prefix = ""
    m_prefix = re.match(r'^(([a-z]+\.\s*)+)1\.\s+', remaining)
    if m_prefix:
        leading_prefix = m_prefix.group(1)
        remaining = remaining[m_prefix.end():]
    else:
        m_one = re.match(r'^1\.\s+', remaining)
        if m_one:
            remaining = remaining[m_one.end():]

    while True:
        next_num = current_num + 1
        pattern_next = re.compile(rf'(?:^|\s+){next_num}\.\s+', re.DOTALL)
        m_next = pattern_next.search(remaining)
        
        if m_next:
            sense_body = remaining[:m_next.start()].strip()
            if current_num == 1 and leading_prefix:
                sense_body = leading_prefix + sense_body
            senses.append((current_num, sense_body))
            remaining = remaining[m_next.end():].strip()
            current_num = next_num
        else:
            sense_body = remaining.strip()
            if current_num == 1 and leading_prefix:
                sense_body = leading_prefix + sense_body
            senses.append((current_num, sense_body))
            break

    return senses


def parse_raw_entry_text(
    headword: str,
    homograph_num: Optional[str],
    body_lines: List[str],
    page_book: int,
    page_pdf: int,
    column: int
) -> Dict[str, Any]:
    """Parse a complete dictionary entry into the canonical digital schema."""
    # Dehyphenate wrapped words across lines within the entry
    if len(body_lines) > 1:
        joined_lines = body_lines[0].strip()
        for next_l in body_lines[1:]:
            next_l = next_l.strip()
            if not next_l:
                continue
            if re.search(r'[a-zA-Zäöüçşg̃ⱬñÄÖÜÇŞG̃Ñ]-$', joined_lines) and re.match(r'^[a-zäöüçşg̃ⱬñ]', next_l):
                joined_lines = joined_lines[:-1] + next_l
            else:
                joined_lines = joined_lines + " " + next_l
        body_text = joined_lines
    else:
        body_text = body_lines[0].strip() if body_lines else ""

    norm_headword = converter.normalize_schwarz_text(headword)

    numbered_senses = split_into_numbered_senses(body_text)
    senses = []
    for s_num, s_body in numbered_senses:
        senses.append(parse_single_sense(s_num, s_body, norm_headword))

    pos_list = []
    domain_list = []

    for s in senses:
        if s.get("pos"):
            pos_list.append(s["pos"])
            s["pos"] = POS_TAGS[s["pos"]]
        if s.get("voice"):
            s["voice"] = VOICE_TAGS[s["voice"]]
        if s.get("domain"):
            domain_list.append(s["domain"])
            s["domain"] = DOMAIN_TAGS[s["domain"]]

    return {
        "headword": {
            "schwarz": norm_headword,
            "uly": converter.custom_to_uly(norm_headword),
            "uey": converter.custom_to_arabic(norm_headword),
            "uyy": converter.custom_to_uyy(norm_headword),
            "ipa": converter.custom_to_ipa(norm_headword)
        },
        "homograph_number": homograph_num,
        "pos": list(dict.fromkeys(pos_list)),
        "domain": list(dict.fromkeys(domain_list)),
        "page_book": page_book,
        "page_pdf": page_pdf,
        "column": column,
        "senses": senses,
        "raw_body": body_text
    }


def parse_column_transcription(
    transcription: str,
    page_book: int,
    page_pdf: int,
    column: int
) -> Tuple[Optional[str], List[Dict[str, Any]], Optional[str]]:
    """Parse transcription of a column into leading continuation and complete entries."""
    lines = [converter.normalize_schwarz_text(l.strip()) for l in transcription.split('\n') if l.strip()]

    leading_continuation = None
    entries = []

    current_headword = None
    current_homograph = None
    current_body = []

    for line in lines:
        if line.startswith('[CONTINUATION]'):
            cont_text = line.replace('[CONTINUATION]', '').strip()
            if cont_text:
                leading_continuation = cont_text
            continue

        # Check if line is a headword
        is_bold_headword = bool(re.match(r'^\*\*[^*]+(?:\*\*(?:\s+[IVXLCDM\d]+)?|\s+[IVXLCDM\d]+\*\*)\s*$', line.strip()))

        if is_bold_headword:
            if current_headword is not None:
                entry = parse_raw_entry_text(
                    current_headword, current_homograph, current_body, page_book, page_pdf, column
                )
                entries.append(entry)

            current_headword, current_homograph = parse_headword_line(line)
            current_body = []
        else:
            if current_headword is None:
                if leading_continuation:
                    leading_continuation += ' ' + line
                else:
                    leading_continuation = line
            else:
                current_body.append(line)

    if current_headword is not None:
        entry = parse_raw_entry_text(
            current_headword, current_homograph, current_body, page_book, page_pdf, column
        )
        entries.append(entry)

    return leading_continuation, entries, None
