from pathlib import Path

from scripts.build_manifest import build_manifest, verify

import json

def test_verify_detects_a_modified_file(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "doc1.md").write_text("contenu original")

    manifest = build_manifest(corpus_dir)
    manifest_path = tmp_path / "manifest.json"

    Path(manifest_path).write_text(json.dumps(manifest, indent=2))

    (corpus_dir / "doc1.md").write_text("test")

    report = verify(manifest_path, corpus_dir)

    assert "doc1.md" in report["modified"]
    assert len(report['missing']) == 0
    assert len(report['added']) == 0


    