from pathlib import Path

import hashlib
import json

URL_TEMPLATE = "https://www.legifrance.gouv.fr/conv_coll/id/{kali_id}/"


def file_sha256(path: Path):
    with open(path, "rb") as file:
        data = file.read()

    return hashlib.sha256(data).hexdigest()


def build_manifest(corpus_dir: Path):
    documents = {}
    corpus_bytes: bytes = b""
    for path in sorted(corpus_dir.glob("*.md")):
        kali_id = path.stem
        documents[path.name] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "source_url": URL_TEMPLATE.format(kali_id=kali_id),
        }
        corpus_bytes += documents[path.name]["sha256"].encode()

    corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()

    return {
        "corpus_hash": corpus_hash,
        "document_count": len(documents),
        "documents": documents,
    }


def verify(manifest_path: Path, corpus_dir: Path):
    manifest = json.loads(manifest_path.read_text())
    recorded = manifest["documents"]

    on_disk = {p.name: p for p in corpus_dir.glob("*.md")}

    recorded_names = set(recorded.keys())
    on_disk_names = set(on_disk.keys())

    missing = sorted(recorded_names - on_disk_names)
    added = sorted(on_disk_names - recorded_names)

    modified = sorted(
        name
        for name in recorded_names & on_disk_names
        if file_sha256(on_disk[name]) != recorded[name]["sha256"]
    )

    return {
        "missing": missing,
        "added": added,
        "modified": modified,
    }


if __name__ == "__main__":
    manifest = build_manifest(Path("data/converted"))
    Path("data/manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"{manifest['document_count']} documents, corpus_hash={manifest['corpus_hash']}"
    )
