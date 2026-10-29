from subtune.align import align
from subtune.srt import clean_text, decode, load_cues, parse_srt

from fake_api import chinese, english, make_srt


def test_parse_srt_reads_times_and_multiline_text():
    text = "1\n00:00:01,500 --> 00:00:03,000\nHello\nthere\n\n2\n00:01:00,000 --> 00:01:02,250\nBye\n"
    cues = parse_srt(text)
    assert [(c.start, c.end, c.text) for c in cues] == [(1.5, 3.0, "Hello\nthere"), (60.0, 62.25, "Bye")]


def test_decode_falls_back_from_non_utf8():
    raw = "1\n00:00:01,000 --> 00:00:02,000\n這是測試字串，看看編碼。\n".encode("big5")
    assert "這是測試" in decode(raw, "zh-tw")


def test_clean_text_strips_tags_hearing_impaired_and_speaker():
    assert clean_text("<i>- JOHN: Hello!</i>\n- [door slams] Hi.", "en") == "Hello! Hi."
    assert clean_text("♪ la la ♪", "en") == "la la"
    assert clean_text("Subtitles by XYZ www.example.com", "en") == ""


def test_clean_text_drops_english_line_of_bilingual_chinese():
    assert clean_text("你好嗎？\nHow are you?", "zh-tw") == "你好嗎？"


def test_align_recovers_offset_and_framerate_drift():
    n = 120
    src = load_cues(make_srt(english(n)).encode(), "en")
    # Target release: 2.7 s late and played at 25/23.976 speed.
    tgt = load_cues(make_srt(chinese(n), offset=2.7, scale=25 / 23.976).encode(), "zh-tw")
    result = align(src, tgt, "en", "zh-tw")
    assert result.ok, result.message
    assert result.coverage > 0.95
    correct = sum(s.split("number ")[1].split(",")[0] == t.split("第")[1].split("句")[0] for s, t, _ in result.pairs)
    assert correct / len(result.pairs) > 0.95


def test_align_rejects_unrelated_files():
    src = load_cues(make_srt(english(80)).encode(), "en")
    tgt = load_cues(make_srt(chinese(25), gap=13.0).encode(), "zh-tw")
    assert not align(src, tgt, "en", "zh-tw").ok
