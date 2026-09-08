import httpx
from pathlib import Path
import time

URL_TEMPLATE = "https://www.legifrance.gouv.fr/conv_coll/id/{kali_id}/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def fetch_one(kali_id: str, dest_dir: Path) -> Path:
    dest = dest_dir / f"{kali_id}.html"

    if dest.exists():
        return dest

    url = URL_TEMPLATE.format(kali_id=kali_id)

    response = httpx.get(url, headers=HEADERS, follow_redirects=True)
    if "Just a moment" in response.text:
        raise RuntimeError(f"Cloudflare challenge blocked fetch for {kali_id}")

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(response.text)
    tmp.rename(dest)

    return dest


def load_kali_ids(path: Path = Path("scripts/kali_ids.txt")):
    return path.read_text().strip().splitlines()


def fetch_corpus(kali_ids: list[str], dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for kali_id in kali_ids:
        dest = dest_dir / f"{kali_id}.html"
        already_cached = dest.exists()
        fetch_one(kali_id, dest_dir)
        if not already_cached:
            time.sleep(1)


if __name__ == "__main__":
    kali_ids = load_kali_ids()
    fetch_corpus(kali_ids, Path("data/raw"))
