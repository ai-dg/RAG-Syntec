from pathlib import Path

from bs4 import BeautifulSoup


def html_to_markdown(html: str, kali_id: str):
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one(".highlightable-content")
    if main is None:
        raise ValueError(f"Could not find main content for {kali_id}")

    title = soup.title.get_text(strip=True) if soup.title else kali_id
    body = main.get_text(separator="\n", strip=True)

    content = title + "\n" + f"{kali_id}" + "\n" + body

    return content


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
