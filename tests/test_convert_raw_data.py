from scripts.convert_raw_data import html_to_markdown

def test_html_to_markdown_extracts_title_and_content():
    fake_html = """
    <html><head><title>Article de test - Légifrance</title></head>
    <body><div class="highlightable-content">Contenu de test</div></body>
    </html>
    """
    result = html_to_markdown(fake_html, "KALITEXT999")
    assert "Article de test" in result
    assert "Contenu de test" in result