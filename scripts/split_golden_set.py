import json
import random
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)

def load_records(path):
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def split_golden_set(records, held_out_fraction=0.25):
    by_class = defaultdict(list)

    for record in records:
        by_class[record["class"]].append(record)

    held_out = []
    visible = []

    for class_name, items in by_class.items():
        random.shuffle(items)
        n_held_out = round(len(items) * held_out_fraction) 
        held_out.extend(items[:n_held_out])
        visible.extend(items[n_held_out:])
    

    return visible, held_out



if __name__ == "__main__":
    records = load_records("eval/golden.jsonl")
    visible, held_out = split_golden_set(records)
    with open("eval/golden.jsonl", "w") as file:
        for record in visible:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open("eval/golden_held_out.jsonl", "w") as file:
        for record in held_out:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    visible_count = Counter(record['class'] for record in visible)
    held_out_count = Counter(record['class'] for record in held_out)

    print("Visible Count:")
    print(f"in_topic_answerable: {visible_count['in_topic_answerable']}")
    print(f"in_topic_unanswerable: {visible_count['in_topic_unanswerable']}")
    print(f"off_topic: {visible_count['off_topic']}")
    print(f"adversarial: {visible_count['adversarial']}")

    print("Held_out Count:")
    print(f"in_topic_answerable: {held_out_count['in_topic_answerable']}")
    print(f"in_topic_unanswerable: {held_out_count['in_topic_unanswerable']}")
    print(f"off_topic: {held_out_count['off_topic']}")
    print(f"adversarial: {held_out_count['adversarial']}")