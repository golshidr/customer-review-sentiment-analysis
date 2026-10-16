from review_sentiment.preprocessing import (
    clean_text,
    has_negation,
    light_clean,
    normalize_for_dedup,
    tokenize_words,
    word_count,
)


def test_clean_text_matches_original_notebook_behaviour():
    raw = "I LOVED it!<br /><br />10/10 -- would watch again."
    assert clean_text(raw) == "i loved it would watch again"


def test_clean_text_handles_non_strings():
    assert clean_text(123) == ""
    assert clean_text(None) == "none"


def test_light_clean_keeps_case_and_punctuation():
    assert light_clean("Great!<br />Not bad.") == "Great! Not bad."


def test_normalize_for_dedup_is_case_and_space_insensitive():
    assert normalize_for_dedup("A  Film<br/>") == normalize_for_dedup("a film")


def test_negation_detection():
    assert has_negation("It was not good")
    assert has_negation("I didn't like it")
    assert not has_negation("It was good")
    assert tokenize_words("Didn't LIKE") == ["didn't", "like"]


def test_word_count_ignores_html():
    assert word_count("one two<br /><br />three") == 3
