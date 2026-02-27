
from customer_retention.generators.notebook_sync.cell_types import (
    CellSyncType,
    detect_cell_sync_type,
    extract_embedded_id,
    extract_tag_name,
    has_magic_comment,
    prepend_magic_comment,
    strip_magic_comment,
)


class TestCellSyncType:

    def test_enum_values(self):
        assert CellSyncType.CONFIG.value == "config"
        assert CellSyncType.USER_CODE.value == "user_code"
        assert CellSyncType.CODE.value == "code"


class TestDetectCellSyncType:

    def test_config_tag(self):
        assert detect_cell_sync_type(["# @cr:config\n", "X = 1\n"]) == CellSyncType.CONFIG

    def test_user_code_tag(self):
        assert detect_cell_sync_type(["# @cr:user_code\n", "def f(): pass\n"]) == CellSyncType.USER_CODE

    def test_code_tag(self):
        assert detect_cell_sync_type(["# @cr:code\n", "import os\n"]) == CellSyncType.CODE

    def test_no_tag_defaults_to_code(self):
        assert detect_cell_sync_type(["import os\n", "print(1)\n"]) == CellSyncType.CODE

    def test_empty_source(self):
        assert detect_cell_sync_type([]) == CellSyncType.CODE

    def test_single_line_no_newline(self):
        assert detect_cell_sync_type(["# @cr:config"]) == CellSyncType.CONFIG

    def test_whitespace_before_tag(self):
        assert detect_cell_sync_type(["  # @cr:config\n"]) == CellSyncType.CODE

    def test_tag_must_be_exact_type(self):
        assert detect_cell_sync_type(["# @cr:configs\n"]) == CellSyncType.CODE
        assert detect_cell_sync_type(["# @cr:user_codes\n"]) == CellSyncType.CODE

    def test_tag_with_name_and_id(self):
        line = "# @cr:code name='load_data' id=a1b2c3d4\n"
        assert detect_cell_sync_type([line]) == CellSyncType.CODE

    def test_config_with_name_and_id(self):
        line = "# @cr:config name='settings' id=f9e8d7c6\n"
        assert detect_cell_sync_type([line]) == CellSyncType.CONFIG

    def test_user_code_with_name_and_id(self):
        line = "# @cr:user_code name='custom_logic' id=b5a4c3d2\n"
        assert detect_cell_sync_type([line]) == CellSyncType.USER_CODE

    def test_tag_with_only_name(self):
        assert detect_cell_sync_type(["# @cr:code name='load_data'\n"]) == CellSyncType.CODE

    def test_tag_with_only_id(self):
        assert detect_cell_sync_type(["# @cr:code id=a1b2c3d4\n"]) == CellSyncType.CODE

    def test_unknown_trailing_text_rejected(self):
        assert detect_cell_sync_type(["# @cr:config extra stuff\n"]) == CellSyncType.CODE


class TestExtractEmbeddedId:

    def test_extracts_id(self):
        assert extract_embedded_id(["# @cr:code name='load' id=a1b2c3d4\n"]) == "a1b2c3d4"

    def test_extracts_id_without_name(self):
        assert extract_embedded_id(["# @cr:code id=a1b2c3d4\n"]) == "a1b2c3d4"

    def test_returns_none_when_no_id(self):
        assert extract_embedded_id(["# @cr:code name='load'\n"]) is None

    def test_returns_none_for_bare_tag(self):
        assert extract_embedded_id(["# @cr:code\n"]) is None

    def test_returns_none_for_no_tag(self):
        assert extract_embedded_id(["import os\n"]) is None

    def test_returns_none_for_empty(self):
        assert extract_embedded_id([]) is None


class TestExtractTagName:

    def test_extracts_name(self):
        assert extract_tag_name(["# @cr:code name='load_data' id=abc\n"]) == "load_data"

    def test_extracts_name_without_id(self):
        assert extract_tag_name(["# @cr:config name='settings'\n"]) == "settings"

    def test_returns_none_when_no_name(self):
        assert extract_tag_name(["# @cr:code id=abc\n"]) is None

    def test_returns_none_for_bare_tag(self):
        assert extract_tag_name(["# @cr:code\n"]) is None

    def test_returns_none_for_no_tag(self):
        assert extract_tag_name(["import os\n"]) is None


class TestHasMagicComment:

    def test_has_config(self):
        assert has_magic_comment(["# @cr:config\n", "X = 1\n"]) is True

    def test_has_user_code(self):
        assert has_magic_comment(["# @cr:user_code\n"]) is True

    def test_has_code(self):
        assert has_magic_comment(["# @cr:code\n"]) is True

    def test_has_tag_with_name_and_id(self):
        assert has_magic_comment(["# @cr:code name='load' id=abc\n"]) is True

    def test_no_magic(self):
        assert has_magic_comment(["import os\n"]) is False

    def test_empty(self):
        assert has_magic_comment([]) is False


class TestStripMagicComment:

    def test_strips_config_tag(self):
        result = strip_magic_comment(["# @cr:config\n", "X = 1\n", "Y = 2\n"])
        assert result == ["X = 1\n", "Y = 2\n"]

    def test_strips_user_code_tag(self):
        result = strip_magic_comment(["# @cr:user_code\n", "def f(): pass\n"])
        assert result == ["def f(): pass\n"]

    def test_strips_tag_with_name_and_id(self):
        result = strip_magic_comment(["# @cr:code name='load' id=abc\n", "run()\n"])
        assert result == ["run()\n"]

    def test_no_tag_returns_unchanged(self):
        lines = ["import os\n", "print(1)\n"]
        assert strip_magic_comment(lines) == lines

    def test_empty_returns_empty(self):
        assert strip_magic_comment([]) == []

    def test_tag_only(self):
        assert strip_magic_comment(["# @cr:config\n"]) == []

    def test_does_not_mutate_input(self):
        original = ["# @cr:config\n", "X = 1\n"]
        copy = list(original)
        strip_magic_comment(original)
        assert original == copy


class TestPrependMagicComment:

    def test_prepend_config(self):
        result = prepend_magic_comment(["X = 1\n"], CellSyncType.CONFIG)
        assert result == ["# @cr:config\n", "X = 1\n"]

    def test_prepend_user_code(self):
        result = prepend_magic_comment(["def f(): pass\n"], CellSyncType.USER_CODE)
        assert result == ["# @cr:user_code\n", "def f(): pass\n"]

    def test_prepend_code(self):
        result = prepend_magic_comment(["import os\n"], CellSyncType.CODE)
        assert result == ["# @cr:code\n", "import os\n"]

    def test_prepend_to_empty(self):
        result = prepend_magic_comment([], CellSyncType.CONFIG)
        assert result == ["# @cr:config\n"]

    def test_does_not_mutate_input(self):
        original = ["X = 1\n"]
        copy = list(original)
        prepend_magic_comment(original, CellSyncType.CONFIG)
        assert original == copy

    def test_replaces_existing_tag(self):
        result = prepend_magic_comment(["# @cr:code\n", "X = 1\n"], CellSyncType.CONFIG)
        assert result == ["# @cr:config\n", "X = 1\n"]

    def test_replaces_tag_with_name_and_id(self):
        result = prepend_magic_comment(["# @cr:code name='x' id=abc\n", "X = 1\n"], CellSyncType.CONFIG)
        assert result == ["# @cr:config\n", "X = 1\n"]
