"""
converter.py - Uyghur Script Conversion Engine

Converts Henry G. Schwarz's custom Latin transcription to:
1. UEY (Uyghur Ereb Yëziqi - Standard Uyghur Arabic Script)
2. ULY (Uyghur Latin Yëziqi - Standard Uyghur Latin Script)
3. UYY (Uyghur Yëngi Yëziq - 1965-1982 Pinyin-based Latin Script)
4. IPA (International Phonetic Alphabet)

Alphabet Mapping:
| Letter | Schwarz | ULY | UEY (Init/Med/Fin) | UYY | IPA |
|--------|---------|-----|--------------------|-----|-----|
| 1      | a       | a   | ئا / ا / ــا       | a   | ɑ   |
| 2      | b       | b   | ب                  | b   | b   |
| 3      | p       | p   | پ                  | p   | p   |
| 4      | t       | t   | ت                  | t   | t   |
| 5      | d       | d   | د                  | d   | d   |
| 6      | ä       | e   | ئە / ە / ــە       | ə   | ɛ   |
| 7      | j       | j   | ج                  | j   | dʒ  |
| 8      | ç       | ch  | چ                  | q   | tʃ  |
| 9      | x       | x   | خ                  | h   | χ   |
| 10     | h       | h   | ھ                  | ⱨ   | h/ɦ |
| 11     | o       | o   | ئو / و / ــو       | o   | o   |
| 12     | ö       | ö   | ئۆ / ۆ / ــۆ       | ɵ   | ø   |
| 13     | s       | s   | س                  | s   | s   |
| 14     | ş       | sh  | ش                  | x   | ʃ   |
| 15     | r       | r   | ر                  | r   | r   |
| 16     | z       | z   | ز                  | z   | z   |
| 17     | ⱬ       | zh  | ژ                  | ⱬ   | ʒ   |
| 18     | w       | w   | ۋ                  | w   | w/v |
| 19     | u       | u   | ئۇ / ۇ / ــۇ       | u   | u   |
| 20     | ü       | ü   | ئۈ / ۈ / ــۈ       | ü   | y   |
| 21     | f       | f   | ف                  | f   | f   |
| 22     | q       | q   | ق                  | ⱪ   | q   |
| 23     | k       | k   | ك                  | k   | k   |
| 24     | ñ       | ng  | ڭ                  | ng  | ŋ   |
| 25     | e       | ë   | ئې / ې / ــې       | e   | e   |
| 26     | i       | i   | ئى / ى / ــى       | i   | i/ɨ |
| 27     | y       | y   | ي                  | y   | j   |
| 28     | g       | g   | گ                  | g   | ɡ   |
| 29     | g̃       | gh  | غ                  | ƣ   | ʁ   |
| 30     | l       | l   | ل                  | l   | l   |
| 31     | m       | m   | م                  | m   | m   |
| 32     | n       | n   | ن                  | n   | n   |
"""

import re
import unicodedata

VOWELS_SCHWARZ = {'a', 'ä', 'o', 'ö', 'u', 'ü', 'e', 'i'}

ARABIC_INITIAL_VOWEL = {
    'a': 'ئا',
    'ä': 'ئە',
    'o': 'ئو',
    'ö': 'ئۆ',
    'u': 'ئۇ',
    'ü': 'ئۈ',
    'e': 'ئې',
    'i': 'ئى',
}

ARABIC_MEDIAL_VOWEL = {
    'a': 'ا',
    'ä': 'ە',
    'o': 'و',
    'ö': 'ۆ',
    'u': 'ۇ',
    'ü': 'ۈ',
    'e': 'ې',
    'i': 'ى',
}

ARABIC_CONSONANTS = {
    'b': 'ب',
    'p': 'پ',
    't': 'ت',
    'j': 'ج',
    'ç': 'چ',
    'x': 'خ',
    'd': 'د',
    'r': 'ر',
    'z': 'ز',
    'ⱬ': 'ژ',
    's': 'س',
    'ş': 'ش',
    'g̃': 'غ',
    'f': 'ف',
    'q': 'ق',
    'k': 'ك',
    'g': 'گ',
    'ñ': 'ڭ',
    'l': 'ل',
    'm': 'م',
    'n': 'ن',
    'h': 'ھ',
    'w': 'ۋ',
    'y': 'ي',
}

ULY_MAP = {
    'a': 'a',
    'ä': 'e',
    'o': 'o',
    'ö': 'ö',
    'u': 'u',
    'ü': 'ü',
    'e': 'ë',
    'i': 'i',
    'b': 'b',
    'p': 'p',
    't': 't',
    'j': 'j',
    'ç': 'ch',
    'x': 'x',
    'd': 'd',
    'r': 'r',
    'z': 'z',
    'ⱬ': 'zh',
    's': 's',
    'ş': 'sh',
    'g̃': 'gh',
    'f': 'f',
    'q': 'q',
    'k': 'k',
    'g': 'g',
    'ñ': 'ng',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'h': 'h',
    'w': 'w',
    'y': 'y',
}

UYY_MAP = {
    'a': 'a',
    'ä': 'ə',
    'o': 'o',
    'ö': 'ɵ',
    'u': 'u',
    'ü': 'ü',
    'e': 'e',
    'i': 'i',
    'b': 'b',
    'p': 'p',
    't': 't',
    'j': 'j',
    'ç': 'q',
    'x': 'h',
    'd': 'd',
    'r': 'r',
    'z': 'z',
    'ⱬ': 'ⱬ',
    's': 's',
    'ş': 'x',
    'g̃': 'ƣ',
    'f': 'f',
    'q': 'ⱪ',
    'k': 'k',
    'ñ': 'ng',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'h': 'ⱨ',
    'w': 'w',
    'y': 'y',
    'g': 'g',
}

IPA_MAP = {
    'a': 'ɑ',
    'ä': 'ɛ',
    'o': 'o',
    'ö': 'ø',
    'u': 'u',
    'ü': 'y',
    'e': 'e',
    'i': 'i',
    'b': 'b',
    'p': 'p',
    't': 't',
    'j': 'dʒ',
    'ç': 'tʃ',
    'x': 'χ',
    'd': 'd',
    'r': 'r',
    'z': 'z',
    'ⱬ': 'ʒ',
    's': 's',
    'ş': 'ʃ',
    'g̃': 'ʁ',
    'f': 'f',
    'q': 'q',
    'k': 'k',
    'g': 'ɡ',
    'ñ': 'ŋ',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'h': 'h',
    'w': 'w',
    'y': 'j',
}


def normalize_schwarz_text(text: str) -> str:
    """Normalize text into canonical Schwarz alphabet representation."""
    if not text:
        return ""
    # Standardize decomposed combining tilde on g
    text = text.replace('g\u0303', 'g̃').replace('G\u0303', 'G̃')
    # Standardize Turkish/Cyrillic OCR misreadings
    text = text.replace('ı', 'i').replace('İ', 'I')
    text = text.replace('ğ', 'g̃').replace('Ğ', 'G̃')
    text = text.replace('ž', 'ⱬ').replace('Ž', 'ⱬ')
    text = text.replace('š', 'ş').replace('Š', 'Ş')
    text = text.replace('č', 'ç').replace('Č', 'Ç')
    text = text.replace('ŋ', 'ñ').replace('Ŋ', 'Ñ')
    text = text.replace('ə', 'ä').replace('Ə', 'Ä')
    return text


def tokenize_schwarz(word: str):
    """Tokenize a single word into character units, keeping 'g̃' and 'G̃' as atomic units."""
    word = normalize_schwarz_text(word)
    tokens = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i:i+2] in ('g̃', 'G̃'):
            tokens.append(word[i:i+2])
            i += 2
        else:
            tokens.append(word[i])
            i += 1
    return tokens


def custom_to_uly(text: str) -> str:
    """Convert Schwarz Latin transcription to standard ULY."""
    if not text:
        return ""
    text = normalize_schwarz_text(text)
    tokens = tokenize_schwarz(text)
    res = []
    for t in tokens:
        tl = t.lower()
        if tl in ULY_MAP:
            m = ULY_MAP[tl]
            res.append(m.upper() if t.isupper() else m)
        else:
            res.append(t)
    return ''.join(res)


def custom_to_uyy(text: str) -> str:
    """Convert Schwarz Latin transcription to 1965-1982 UYY (Yëngi Yëziq)."""
    if not text:
        return ""
    text = normalize_schwarz_text(text)
    tokens = tokenize_schwarz(text)
    res = []
    for t in tokens:
        tl = t.lower()
        if tl in UYY_MAP:
            m = UYY_MAP[tl]
            res.append(m.upper() if t.isupper() else m)
        else:
            res.append(t)
    return ''.join(res)


def custom_to_ipa(text: str) -> str:
    """Convert Schwarz Latin transcription to approximate IPA."""
    if not text:
        return ""
    text = normalize_schwarz_text(text)
    tokens = tokenize_schwarz(text)
    res = []
    for t in tokens:
        tl = t.lower()
        if tl in IPA_MAP:
            res.append(IPA_MAP[tl])
        else:
            res.append(t)
    return ''.join(res)


def custom_to_arabic_word(word: str) -> str:
    """Convert a single Uyghur word in Schwarz Latin transcription to Uyghur Arabic script (UEY)."""
    if not word:
        return ""
    
    # Strip any punctuation attached to word for conversion and reattach
    prefix_punct = ""
    suffix_punct = ""
    
    # Check leading non-alphabet chars
    m_pre = re.match(r"^([^a-zA-Zäöüçşg̃ⱬñÄÖÜÇŞG̃Ñ'’\-]+)(.*)", word)
    if m_pre:
        prefix_punct = m_pre.group(1)
        word = m_pre.group(2)
        
    m_suf = re.search(r"([^a-zA-Zäöüçşg̃ⱬñÄÖÜÇŞG̃Ñ'’\-]+)$", word)
    if m_suf:
        suffix_punct = m_suf.group(1)
        word = word[:-len(suffix_punct)]

    if not word:
        return prefix_punct + suffix_punct

    tokens = tokenize_schwarz(word)
    res = []
    is_syllable_start = True
    prev_was_vowel = False

    for t in tokens:
        tl = t.lower()
        if tl in VOWELS_SCHWARZ:
            if is_syllable_start or prev_was_vowel:
                res.append(ARABIC_INITIAL_VOWEL[tl])
            else:
                res.append(ARABIC_MEDIAL_VOWEL[tl])
            prev_was_vowel = True
            is_syllable_start = False
        elif tl in ARABIC_CONSONANTS:
            res.append(ARABIC_CONSONANTS[tl])
            prev_was_vowel = False
            is_syllable_start = False
        elif t in ("'", "’"):
            # Glottal stop / syllable separator
            res.append('ئ')
            prev_was_vowel = False
            is_syllable_start = False
        elif t == '-':
            res.append('-')
            is_syllable_start = True
            prev_was_vowel = False
        else:
            res.append(t)
            is_syllable_start = True
            prev_was_vowel = False

    return prefix_punct + ''.join(res) + suffix_punct


def custom_to_arabic(text: str) -> str:
    """Convert full text (words, sentences, phrases) to Uyghur Arabic script (UEY)."""
    if not text:
        return ""
    text = normalize_schwarz_text(text)
    # Split preserving whitespace
    parts = re.split(r'(\s+)', text)
    converted = []
    for p in parts:
        if p.isspace() or not p:
            converted.append(p)
        else:
            converted.append(custom_to_arabic_word(p))
    return ''.join(converted)
