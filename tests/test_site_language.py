from scripts.site_language import detect_page_language


def test_html_lang_detects_page_version():
    html = '<html lang="ru-RU"><head><title>Сайт</title></head><body>Привет</body></html>'
    assert detect_page_language(html, "https://example.com/") == "ru"


def test_url_locale_overrides_stale_html_lang():
    html = '<html lang="en"><body>Русская версия сайта</body></html>'
    assert detect_page_language(html, "https://example.com/ru/services") == "ru"
    assert detect_page_language(html, "https://example.com/?lang=ru") == "ru"
    assert detect_page_language(html, "https://example.com/#/ru/services") == "ru"
    assert detect_page_language(html, "https://example.com/#!/ru/services") == "ru"
    assert detect_page_language("<html><body>Services</body></html>", "https://example.com/en-guide") is None


def test_metadata_detects_page_language():
    html = '<html><head><meta property="og:locale" content="kk_KZ"></head></html>'
    assert detect_page_language(html, "https://example.com/") == "kk"


def test_visible_cyrillic_fallback_ignores_script_content():
    script = '<script>' + ('Русский текст ' * 30) + '</script>'
    assert detect_page_language(f'<html><body>{script}Hello world</body></html>', "https://example.com/") is None
    assert detect_page_language('<html><body>' + ('Русский текст ' * 30) + '</body></html>', "https://example.com/") == "ru"
