import re
from pathlib import Path
import httpx

INDEX_URL = "https://www.legifrance.gouv.fr/conv_coll/1486"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def discover_kali_ids():
    response = httpx.get(INDEX_URL, headers=HEADERS, follow_redirects=True)
    ids = re.findall(r"KALITEXT\d+", response.text)
    ids = sorted(set(ids))
    return ids


if __name__ == "__main__":
    ids = discover_kali_ids()
    print(f"{len(ids)} identifiers found")
    Path("scripts/kali_ids.txt").write_text("\n".join(ids) + "\n")
