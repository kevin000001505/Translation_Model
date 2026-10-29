"""Languages offered in the UI, and how each model family names them.

Codes are OpenSubtitles language codes (from GET /infos/languages). Chinese is offered only as
Traditional (zh-tw); Simplified, Cantonese and the "Chinese bilingual" pseudo-language are left out.
"""

from __future__ import annotations

# (OpenSubtitles code, display name), as returned by /infos/languages.
_OPENSUBTITLES_LANGUAGES: list[tuple[str, str]] = [
    ("ab", "Abkhazian"),
    ("af", "Afrikaans"),
    ("sq", "Albanian"),
    ("am", "Amharic"),
    ("ar", "Arabic"),
    ("an", "Aragonese"),
    ("hy", "Armenian"),
    ("as", "Assamese"),
    ("at", "Asturian"),
    ("az-az", "Azerbaijani"),
    ("eu", "Basque"),
    ("be", "Belarusian"),
    ("bn", "Bengali"),
    ("bs", "Bosnian"),
    ("br", "Breton"),
    ("bg", "Bulgarian"),
    ("my", "Burmese"),
    ("ca", "Catalan"),
    ("ze", "Chinese bilingual"),
    ("zh-ca", "Chinese (Cantonese)"),
    ("zh-cn", "Chinese (simplified)"),
    ("zh-tw", "Chinese (traditional)"),
    ("hr", "Croatian"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("pr", "Dari"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("eo", "Esperanto"),
    ("et", "Estonian"),
    ("ex", "Extremaduran"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("gd", "Gaelic"),
    ("gl", "Galician"),
    ("ka", "Georgian"),
    ("de", "German"),
    ("el", "Greek"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("is", "Icelandic"),
    ("ig", "Igbo"),
    ("id", "Indonesian"),
    ("ia", "Interlingua"),
    ("ga", "Irish"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("kn", "Kannada"),
    ("kk", "Kazakh"),
    ("km", "Khmer"),
    ("ko", "Korean"),
    ("ku", "Kurdish"),
    ("lv", "Latvian"),
    ("lt", "Lithuanian"),
    ("lb", "Luxembourgish"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("ml", "Malayalam"),
    ("ma", "Manipuri"),
    ("mr", "Marathi"),
    ("mn", "Mongolian"),
    ("me", "Montenegrin"),
    ("nv", "Navajo"),
    ("ne", "Nepali"),
    ("se", "Northern Sami"),
    ("no", "Norwegian"),
    ("oc", "Occitan"),
    ("or", "Odia"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt-pt", "Portuguese"),
    ("pt-br", "Portuguese (BR)"),
    ("pm", "Portuguese (MZ)"),
    ("ps", "Pushto"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sx", "Santali"),
    ("sr", "Serbian"),
    ("sd", "Sindhi"),
    ("si", "Sinhalese"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("so", "Somali"),
    ("az-zb", "South Azerbaijani"),
    ("es", "Spanish"),
    ("sp", "Spanish (EU)"),
    ("ea", "Spanish (LA)"),
    ("sw", "Swahili"),
    ("sv", "Swedish"),
    ("sy", "Syriac"),
    ("tl", "Tagalog"),
    ("ta", "Tamil"),
    ("tt", "Tatar"),
    ("te", "Telugu"),
    ("tm-td", "Tetum"),
    ("th", "Thai"),
    ("tp", "Toki Pona"),
    ("tr", "Turkish"),
    ("tk", "Turkmen"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("uz", "Uzbek"),
    ("vi", "Vietnamese"),
    ("cy", "Welsh"),
]

_EXCLUDED = {"ze", "zh-ca", "zh-cn"}
_DISPLAY_OVERRIDES = {"zh-tw": "Chinese (Traditional)", "pt-pt": "Portuguese (PT)"}

LANGUAGES: dict[str, str] = {
    code: _DISPLAY_OVERRIDES.get(code, name)
    for code, name in _OPENSUBTITLES_LANGUAGES
    if code not in _EXCLUDED
}

# OpenSubtitles uses a few codes that are not ISO 639-1. Models expect ISO codes, with an optional
# ISO 3166 region ("pt-BR"). Anything not listed here is already ISO 639-1.
_ISO_CODES = {
    "zh-tw": "zh-TW",
    "pt-pt": "pt-PT",
    "pt-br": "pt-BR",
    "pm": "pt-MZ",
    "sp": "es-ES",
    "ea": "es",
    "az-az": "az",
    "az-zb": "azb",
    "at": "ast",
    "ma": "mni",
    "pr": "prs",
    "sx": "sat",
    "ex": "ext",
    "tp": "tok",
    "me": "cnr",
    "tm-td": "tet",
}

# English names used inside instruction prompts.
_PROMPT_NAMES = {"zh-tw": "Traditional Chinese (Taiwan)", "pt-br": "Brazilian Portuguese"}

# Hunyuan-MT prompts that involve Chinese are written in Chinese and name the target language in
# Chinese (from the Hunyuan-MT-7B model card).
HUNYUAN_CHINESE_NAMES = {
    "zh-tw": "繁体中文",
    "en": "英语",
    "fr": "法语",
    "pt-pt": "葡萄牙语",
    "pt-br": "葡萄牙语",
    "es": "西班牙语",
    "ja": "日语",
    "tr": "土耳其语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "ko": "韩语",
    "th": "泰语",
    "it": "意大利语",
    "de": "德语",
    "vi": "越南语",
    "ms": "马来语",
    "id": "印尼语",
    "tl": "菲律宾语",
    "hi": "印地语",
    "pl": "波兰语",
    "cs": "捷克语",
    "nl": "荷兰语",
    "km": "高棉语",
    "my": "缅甸语",
    "fa": "波斯语",
    "ur": "乌尔都语",
    "te": "泰卢固语",
    "mr": "马拉地语",
    "he": "希伯来语",
    "bn": "孟加拉语",
    "ta": "泰米尔语",
    "uk": "乌克兰语",
    "kk": "哈萨克语",
    "mn": "蒙古语",
}

# Scripts whose characters we can check for, to drop lines in the wrong language (e.g. the English
# line in a bilingual Chinese subtitle).
CJK_LANGUAGES = {"zh-tw", "ja", "ko"}


def iso_code(code: str) -> str:
    return _ISO_CODES.get(code, code)


def base_code(code: str) -> str:
    """ISO 639 code without region, e.g. zh-tw -> zh."""
    return iso_code(code).split("-")[0].lower()


def prompt_name(code: str) -> str:
    return _PROMPT_NAMES.get(code, LANGUAGES.get(code, code))


def pair_key(src: str, tgt: str) -> str:
    return f"{src}__{tgt}"


def choices() -> list[tuple[str, str]]:
    """(label, value) pairs for a dropdown, sorted by name."""
    return sorted(((f"{name} ({code})", code) for code, name in LANGUAGES.items()), key=lambda c: c[0])
