from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from customer_retention.analysis.auto_explorer.skip_logic import (
    TEMPORAL_NOTEBOOKS,
    TEXT_NOTEBOOKS,
    detect_global_skip_set,
    detect_skip_set_for_dataset,
    find_dataset_findings,
    has_text_columns_for_dataset,
    is_event_level_dataset,
)


@pytest.fixture()
def findings_dir(tmp_path):
    d = tmp_path / "findings"
    d.mkdir()
    return d


def _write_findings(path: Path, *, granularity="entity_level", text_processing=None, columns=None, metadata=None):
    data = {
        "source_path": "/data/test.csv",
        "source_format": "csv",
        "row_count": 100,
        "column_count": 5,
        "columns": columns or {
            "id": {"name": "id", "inferred_type": "identifier", "confidence": 0.9, "evidence": []},
        },
        "metadata": metadata or {},
    }
    if granularity == "event_level":
        data["time_series_metadata"] = {
            "granularity": "event_level",
            "temporal_pattern": "regular",
            "entity_column": "customer_id",
            "time_column": "event_date",
        }
    if text_processing:
        data["text_processing"] = text_processing
    path.write_text(yaml.dump(data))


class TestFindDatasetFindings:
    def test_exact_match(self, findings_dir):
        path = findings_dir / "orders_findings.yaml"
        path.write_text("test: true")
        assert find_dataset_findings(findings_dir, "orders") == path

    def test_aggregated_fallback(self, findings_dir):
        path = findings_dir / "orders_aggregated_findings.yaml"
        path.write_text("test: true")
        assert find_dataset_findings(findings_dir, "orders") == path

    def test_glob_fallback(self, findings_dir):
        path = findings_dir / "orders_v2_findings.yaml"
        path.write_text("test: true")
        assert find_dataset_findings(findings_dir, "orders") == path

    def test_no_match(self, findings_dir):
        assert find_dataset_findings(findings_dir, "missing") is None

    def test_nonexistent_dir(self, tmp_path):
        assert find_dataset_findings(tmp_path / "nope", "orders") is None

    def test_namespace_priority(self, findings_dir, tmp_path):
        findings_dir_file = findings_dir / "orders_findings.yaml"
        findings_dir_file.write_text("from_dir: true")

        ns_dir = tmp_path / "ns_datasets" / "orders" / "findings"
        ns_dir.mkdir(parents=True)
        ns_file = ns_dir / "orders_findings.yaml"
        ns_file.write_text("from_ns: true")

        ns = SimpleNamespace(dataset_findings_dir=lambda name: ns_dir)
        assert find_dataset_findings(findings_dir, "orders", namespace=ns) == ns_file


class TestIsEventLevelDataset:
    def test_event_from_findings(self, findings_dir):
        _write_findings(findings_dir / "events_findings.yaml", granularity="event_level")
        assert is_event_level_dataset(findings_dir, "events") is True

    def test_entity_from_findings(self, findings_dir):
        _write_findings(findings_dir / "profiles_findings.yaml", granularity="entity_level")
        assert is_event_level_dataset(findings_dir, "profiles") is False

    def test_from_context_relationship(self, findings_dir):
        entry = SimpleNamespace(relationship="many_to_one")
        context = SimpleNamespace(datasets={"orders": entry})
        assert is_event_level_dataset(findings_dir, "orders", context=context) is True

    def test_default_false(self, findings_dir):
        assert is_event_level_dataset(findings_dir, "unknown") is False


class TestHasTextColumnsForDataset:
    def test_text_from_findings(self, findings_dir):
        columns = {
            "id": {"name": "id", "inferred_type": "identifier", "confidence": 0.9, "evidence": []},
            "notes": {"name": "notes", "inferred_type": "text", "confidence": 0.8, "evidence": []},
        }
        _write_findings(findings_dir / "tickets_findings.yaml", columns=columns)
        assert has_text_columns_for_dataset(findings_dir, "tickets") is True

    def test_text_from_text_processing(self, findings_dir):
        _write_findings(
            findings_dir / "tickets_findings.yaml",
            text_processing={
                "notes": {
                    "column_name": "notes",
                    "embedding_model": "tfidf",
                    "embedding_dim": 100,
                    "n_components": 10,
                    "explained_variance": 0.9,
                    "component_columns": ["notes_0", "notes_1"],
                    "variance_threshold_used": 0.01,
                },
            },
        )
        assert has_text_columns_for_dataset(findings_dir, "tickets") is True

    def test_no_text(self, findings_dir):
        _write_findings(findings_dir / "profiles_findings.yaml")
        assert has_text_columns_for_dataset(findings_dir, "profiles") is False

    def test_auto_drop_text(self, findings_dir):
        columns = {
            "notes": {"name": "notes", "inferred_type": "text", "confidence": 0.8, "evidence": []},
        }
        _write_findings(
            findings_dir / "tickets_findings.yaml",
            columns=columns,
            metadata={"auto_drop_text_columns": True},
        )
        assert has_text_columns_for_dataset(findings_dir, "tickets") is False

    def test_no_findings(self, findings_dir):
        assert has_text_columns_for_dataset(findings_dir, "missing") is False


class TestDetectSkipSetForDataset:
    def test_entity_skips_temporal(self, findings_dir):
        _write_findings(findings_dir / "profiles_findings.yaml", granularity="entity_level")
        skip, reasons = detect_skip_set_for_dataset(findings_dir, "profiles")
        assert TEMPORAL_NOTEBOOKS <= skip
        for nb in TEMPORAL_NOTEBOOKS - TEXT_NOTEBOOKS:
            assert "event-level" in reasons[nb]

    def test_event_no_temporal_skip(self, findings_dir):
        _write_findings(findings_dir / "events_findings.yaml", granularity="event_level")
        skip, _ = detect_skip_set_for_dataset(findings_dir, "events")
        temporal_only = TEMPORAL_NOTEBOOKS - TEXT_NOTEBOOKS
        assert not (temporal_only & skip)

    def test_no_text_skips_text_notebooks(self, findings_dir):
        _write_findings(findings_dir / "profiles_findings.yaml")
        skip, reasons = detect_skip_set_for_dataset(findings_dir, "profiles")
        assert TEXT_NOTEBOOKS <= skip
        for nb in TEXT_NOTEBOOKS:
            assert "TEXT" in reasons[nb]

    def test_with_text_no_text_skip(self, findings_dir):
        columns = {
            "notes": {"name": "notes", "inferred_type": "text", "confidence": 0.8, "evidence": []},
        }
        _write_findings(findings_dir / "tickets_findings.yaml", columns=columns)
        skip, reasons = detect_skip_set_for_dataset(findings_dir, "tickets")
        for nb in TEXT_NOTEBOOKS:
            if nb in skip:
                assert "TEXT" not in reasons[nb]

    def test_event_with_text_no_skip(self, findings_dir):
        columns = {
            "id": {"name": "id", "inferred_type": "identifier", "confidence": 0.9, "evidence": []},
            "notes": {"name": "notes", "inferred_type": "text", "confidence": 0.8, "evidence": []},
        }
        _write_findings(
            findings_dir / "events_findings.yaml",
            granularity="event_level",
            columns=columns,
        )
        skip, _ = detect_skip_set_for_dataset(findings_dir, "events")
        assert len(skip) == 0


class TestDetectGlobalSkipSet:
    def test_any_text_across_datasets(self, findings_dir):
        _write_findings(findings_dir / "profiles_findings.yaml")
        columns = {
            "notes": {"name": "notes", "inferred_type": "text", "confidence": 0.8, "evidence": []},
        }
        _write_findings(findings_dir / "tickets_findings.yaml", columns=columns)

        context = SimpleNamespace(datasets={
            "profiles": SimpleNamespace(path=None),
            "tickets": SimpleNamespace(path=None),
        })
        skip, _ = detect_global_skip_set(findings_dir, context)
        assert len(skip) == 0

    def test_no_text_in_any_dataset(self, findings_dir):
        _write_findings(findings_dir / "profiles_findings.yaml")
        _write_findings(findings_dir / "accounts_findings.yaml")

        context = SimpleNamespace(datasets={
            "profiles": SimpleNamespace(path=None),
            "accounts": SimpleNamespace(path=None),
        })
        skip, _ = detect_global_skip_set(findings_dir, context)
        # _GLOBAL_TEXT_NOTEBOOKS is empty, so no skip entries expected
        assert len(skip) == 0
