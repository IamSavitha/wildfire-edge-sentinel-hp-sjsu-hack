import pytest
from pydantic import ValidationError

from sentinel.schema import CONTEXT_JSON_SCHEMA, ContextResult, Detection, Severity


def test_detection_area():
    assert Detection("smoke", 0.9, (10, 20, 30, 60)).area == 800


def test_severity_ordering():
    assert Severity.ALERT > Severity.MONITOR > Severity.LOG > Severity.IGNORE


def test_context_rejects_unknown_source():
    with pytest.raises(ValidationError):
        ContextResult(source_type="volcano", smoke_color="grey", attended="no",
                      near_structures=False, near_road=False, size_estimate="small",
                      description="x")


def test_json_schema_constrains_source_type():
    enum = CONTEXT_JSON_SCHEMA["properties"]["source_type"]["enum"]
    assert "campfire" in enum and "fog_dust_cloud" in enum
