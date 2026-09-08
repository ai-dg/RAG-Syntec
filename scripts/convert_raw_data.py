from pathlib import Path

from bs4 import BeautifulSoup


def html_to_markdown(html: str, kali_id: str):
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.title.get_text(strip=True) if soup.title else kali_id

    lines = [f"# {title_tag}", "", f"> Identifiant Légifrance : `{kali_id}`", ""]

    articles = soup.select("article.list-article-consommation")

    for article in articles:
        title = article.select_one(".name-article")
        if title:
            title = title.get_text(strip=True)
        else:
            continue
        content = article.select_one(".content")
        if content:
            content = content.get_text(separator="\n", strip=True)
        else:
            continue

        lines.extend([f"## {title}", "", content, ""])


    return "\n".join(lines)

def convert_all(raw_dir: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for html_file in sorted(raw_dir.glob("*.html")):
        kali_id = html_file.stem
        html_content = html_file.read_text()
        content = html_to_markdown(html_content, kali_id)

        dest = dest_dir / f"{kali_id}.md"
        dest.write_text(content)


if __name__ == "__main__":
    convert_all(Path("data/raw"), Path("data/converted"))
