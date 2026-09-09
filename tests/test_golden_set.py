import json
from collections import Counter
from pathlib import Path

import jsonschema

GOLDEN_PATH = Path("eval/golden.jsonl")
HELD_OUT_PATH = Path("eval/golden_held_out.jsonl")
SCHEMA_PATH = Path("eval/schema.json")

def load_records():
    schema = json.loads(SCHEMA_PATH.read_text())
    records = []
    with open(GOLDEN_PATH) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            jsonschema.validate(record, schema)
            records.append(record)

    with open(HELD_OUT_PATH) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            jsonschema.validate(record, schema)
            records.append(record)
    
    return records

def test_every_record_is_schema_valid():
    records = load_records()
    assert len(records) > 0


def test_no_duplicate_ids():
    records = load_records()
    ids = [r["id"] for r in records]

    counter = {}

    for id in ids:
        index = counter.get(id, 0)
        counter[id] = index + 1

        assert counter[id] == 1



def test_class_counts_meet_targets():
    records = load_records()
    counts = Counter(r["class"] for r in records)

    assert counts['in_topic_answerable'] >= 25
    assert counts['in_topic_unanswerable'] >= 10
    assert counts['off_topic'] >= 10
    assert counts['adversarial'] >= 10
    
