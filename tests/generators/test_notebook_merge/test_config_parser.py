
from customer_retention.generators.notebook_merge.config_parser import (
    AssignmentKind,
    parse_config_cell,
)


class TestParseScalars:

    def test_integer(self):
        source = "# @cr:config name='test' id=aabb1122\nPREDICTION_HORIZON = 90\n"
        parsed = parse_config_cell(source)
        assert parsed.assignments["PREDICTION_HORIZON"].kind == AssignmentKind.SCALAR
        assert parsed.assignments["PREDICTION_HORIZON"].literal_value == 90

    def test_string(self):
        source = '# @cr:config name="t" id=aabb1122\nPROJECT_NAME = "email"\n'
        parsed = parse_config_cell(source)
        assert parsed.assignments["PROJECT_NAME"].literal_value == "email"

    def test_boolean(self):
        source = "# @cr:config name='t' id=aabb1122\nLIGHT_RUN = False\n"
        parsed = parse_config_cell(source)
        assert parsed.assignments["LIGHT_RUN"].literal_value is False

    def test_none(self):
        source = "# @cr:config name='t' id=aabb1122\nTARGET_COLUMN = None\n"
        parsed = parse_config_cell(source)
        assert parsed.assignments["TARGET_COLUMN"].kind == AssignmentKind.SCALAR
        assert parsed.assignments["TARGET_COLUMN"].literal_value is None


class TestParseLists:

    def test_simple_list(self):
        source = "# @cr:config name='t' id=aabb1122\nDROP_COLUMNS = [\"body\", \"raw_html\"]\n"
        parsed = parse_config_cell(source)
        assert parsed.assignments["DROP_COLUMNS"].kind == AssignmentKind.LIST
        assert parsed.assignments["DROP_COLUMNS"].literal_value == ["body", "raw_html"]

    def test_empty_list(self):
        source = "# @cr:config name='t' id=aabb1122\nEXCLUDE_DATASETS = []\n"
        parsed = parse_config_cell(source)
        assert parsed.assignments["EXCLUDE_DATASETS"].kind == AssignmentKind.LIST
        assert parsed.assignments["EXCLUDE_DATASETS"].literal_value == []

    def test_list_of_tuples(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            'MILESTONE_PAIRS = [("CREATED", "CLOSED"), ("CREATED", "RESPONSE")]\n'
        )
        parsed = parse_config_cell(source)
        assert parsed.assignments["MILESTONE_PAIRS"].kind == AssignmentKind.LIST
        assert ("CREATED", "CLOSED") in parsed.assignments["MILESTONE_PAIRS"].literal_value


class TestParseDicts:

    def test_simple_dict(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            'DROP_COLUMNS = {"emails": ["body"], "transactions": []}\n'
        )
        parsed = parse_config_cell(source)
        assert parsed.assignments["DROP_COLUMNS"].kind == AssignmentKind.DICT
        assert parsed.assignments["DROP_COLUMNS"].literal_value == {
            "emails": ["body"], "transactions": [],
        }

    def test_nested_dict(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            'TYPE_OVERRIDES = {"emails": {"amount": "numeric"}}\n'
        )
        parsed = parse_config_cell(source)
        assert parsed.assignments["TYPE_OVERRIDES"].kind == AssignmentKind.DICT
        assert parsed.assignments["TYPE_OVERRIDES"].literal_value["emails"]["amount"] == "numeric"

    def test_empty_dict(self):
        source = "# @cr:config name='t' id=aabb1122\nTYPE_OVERRIDES = {}\n"
        parsed = parse_config_cell(source)
        assert parsed.assignments["TYPE_OVERRIDES"].kind == AssignmentKind.DICT
        assert parsed.assignments["TYPE_OVERRIDES"].literal_value == {}


class TestParseComplex:

    def test_enum_ref(self):
        source = "# @cr:config name='t' id=aabb1122\nTEMPORAL_POSTURE = TemporalPosture.STABLE\n"
        parsed = parse_config_cell(source)
        assert parsed.assignments["TEMPORAL_POSTURE"].kind == AssignmentKind.COMPLEX
        assert parsed.assignments["TEMPORAL_POSTURE"].literal_value is None

    def test_function_call(self):
        source = "# @cr:config name='t' id=aabb1122\nX = some_function(arg1, arg2)\n"
        parsed = parse_config_cell(source)
        assert parsed.assignments["X"].kind == AssignmentKind.COMPLEX


class TestPreambleAndStructure:

    def test_tag_line_extracted(self):
        source = "# @cr:config name='discovery_config' id=e5b9a57b\nX = 1\n"
        parsed = parse_config_cell(source)
        assert parsed.tag_line == "# @cr:config name='discovery_config' id=e5b9a57b"

    def test_preamble_with_imports(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            "from customer_retention.foo import Bar\n"
            "\n"
            "X = 1\n"
        )
        parsed = parse_config_cell(source)
        assert "from customer_retention.foo import Bar" in parsed.preamble
        assert "X" in parsed.assignments

    def test_preamble_with_try_except(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            "try:\n"
            '    _widget = dbutils.widgets.get("x")\n'
            "except Exception:\n"
            '    _widget = ""\n'
            "\n"
            "X = 1\n"
        )
        parsed = parse_config_cell(source)
        assert "dbutils" in parsed.preamble
        assert "X" in parsed.assignments

    def test_multiple_assignments(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            "X = 1\n"
            "Y = \"hello\"\n"
            "Z = [1, 2, 3]\n"
        )
        parsed = parse_config_cell(source)
        assert len(parsed.assignments) == 3
        assert parsed.assignments["X"].kind == AssignmentKind.SCALAR
        assert parsed.assignments["Y"].kind == AssignmentKind.SCALAR
        assert parsed.assignments["Z"].kind == AssignmentKind.LIST

    def test_parse_succeeded(self):
        source = "# @cr:config name='t' id=aabb1122\nX = 1\n"
        assert parse_config_cell(source).parse_succeeded is True

    def test_parse_failure_on_invalid_python(self):
        source = "# @cr:config name='t' id=aabb1122\nX = {\n"
        parsed = parse_config_cell(source)
        assert parsed.parse_succeeded is False
        assert len(parsed.assignments) == 0

    def test_trailing_content(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            "X = 1\n"
            "# --- end of config ---\n"
        )
        parsed = parse_config_cell(source)
        assert "end of config" in parsed.trailing

    def test_raw_source_roundtrip(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            'DROP_COLUMNS = {"emails": ["body", "raw_html"]}\n'
        )
        parsed = parse_config_cell(source)
        assert "DROP_COLUMNS" in parsed.assignments["DROP_COLUMNS"].raw_source

    def test_comments_between_assignments_in_raw_source(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            "# prediction horizon in days\n"
            "PREDICTION_HORIZON = 90\n"
            "# recent window\n"
            "RECENT_WINDOW_DAYS = 365\n"
        )
        parsed = parse_config_cell(source)
        assert "prediction horizon" in parsed.assignments["PREDICTION_HORIZON"].raw_source
        assert "recent window" in parsed.assignments["RECENT_WINDOW_DAYS"].raw_source

    def test_multiline_assignment(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            "datasets = {\n"
            '    "customer_emails": "../tests/fixtures/customer_emails.csv"\n'
            "}\n"
        )
        parsed = parse_config_cell(source)
        assert "datasets" in parsed.assignments
        assert parsed.assignments["datasets"].kind == AssignmentKind.DICT

    def test_underscore_prefixed_assignment_in_preamble(self):
        source = (
            "# @cr:config name='t' id=aabb1122\n"
            "try:\n"
            '    _widget_path = dbutils.widgets.get("data_path")\n'
            "except Exception:\n"
            '    _widget_path = ""\n'
            "\n"
            "DATA_PATH = None\n"
        )
        parsed = parse_config_cell(source)
        assert "DATA_PATH" in parsed.assignments
        assert "_widget_path" not in parsed.assignments
