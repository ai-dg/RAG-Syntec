from scripts.convert_raw_data import html_to_markdown

def test_html_to_markdown_extracts_title_and_content():
    fake_html = """
    <html><head><title>Article de test - Légifrance</title></head>
    <body>
    <article class="list-article-consommation">
        <p class="name-article">Article 1er</p>
        <div class="content">Contenu de test</div>
    </article>
    </body></html>
    """
    result = html_to_markdown(fake_html, "KALITEXT999")
    assert "Article de test" in result
    assert "Contenu de test" in result