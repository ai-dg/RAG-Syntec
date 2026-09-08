import json
import jsonschema.exceptions
import pytest
import jsonschema

def test_schema_itself_is_valid():
    schema = json.loads(open("eval/schema.json").read())

    jsonschema.Draft202012Validator.check_schema(schema)


def test_valid_example_passes():
    schema = json.loads(open("eval/schema.json").read())

    example = {
        "id": "q001",
        "question": "Quel est le préavis de démission pour un cadre ?",
        "class": "in_topic_answerable",
        "relevant_chunk_ids": ["KALITEXT000005679895#chunk_7"],
        "expected_answer_gist": "...",
        "notes": "...",
        "corpus_hash": "ec3aa954b89d8e85facb9ce8ebafd209fa94bb6fab7108993fbe79847cc0b4fa",
    }

    jsonschema.validate(example, schema)

def test_invalid_class_is_rejected():
    schema = json.loads(open("eval/schema.json").read())

    example = {
        "id": "q001",
        "question": "...",
        "class": "hors_sujet",
        "relevant_chunk_ids": [],
        "expected_answer_gist": "...",
        "notes": "...",
        "corpus_hash": "...",
    }

    with pytest.raises(jsonschema.exceptions.ValidationError):
        jsonschema.validate(example, schema)

