from scripts.rag_core import detect_query_language, resolve_response_language


def test_detects_russian_query_with_english_project_name():
    assert detect_query_language("Расскажи, как работает проект QLoRA") == "ru"


def test_detects_kazakh_before_generic_cyrillic():
    assert detect_query_language("Бұл жоба қалай жұмыс істейді?") == "kk"


def test_detects_ukrainian_specific_letters():
    assert detect_query_language("Як працює цей проєкт?") == "uk"


def test_does_not_guess_language_from_short_or_latin_text():
    assert detect_query_language("?") is None
    assert detect_query_language("How does WRS work?") is None


def test_explicit_widget_language_has_highest_priority():
    assert resolve_response_language("Расскажи о проекте", "en", "ru") == "en"


def test_detected_query_language_overrides_site_default():
    assert resolve_response_language("Расскажи о проекте", None, "en") == "ru"


def test_site_language_is_used_when_detection_is_inconclusive():
    assert resolve_response_language("How does WRS work?", None, "en-US") == "en"


def test_unsupported_explicit_language_does_not_disable_detection():
    assert resolve_response_language("Что умеет ассистент?", "xx", "en") == "ru"
