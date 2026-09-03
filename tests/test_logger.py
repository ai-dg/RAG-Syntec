import json
import logging
from app.logger import JsonFormatter

def test_json_formatter_produces_parseable_json_with_expected_keys():
    formatter = JsonFormatter()

    record = logging.LogRecord(
        name="test.logger", 
        level=logging.INFO, 
        pathname="test", 
        lineno=0, 
        msg="Hello World", 
        args=(), 
        exc_info=None
    )
    
    output = formatter.format(record)

    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test.logger"
    assert parsed["message"] == "Hello World"
    assert "timestamp" in parsed

def test_json_formatter_includes_extra_fields():
    formatter = JsonFormatter()

    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="test",
        lineno=0,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.request_id = "abc-123"

    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["request_id"] == "abc-123"