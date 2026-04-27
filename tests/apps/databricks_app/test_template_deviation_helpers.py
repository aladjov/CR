from src.template import HELPERS, DataSource, _parse_frontmatter


def _call(name, *args):
    return HELPERS[name](None, *args)


def test_dev_bar_pct_zero_for_missing():
    assert _call("dev_bar_pct", None) == "0"
    assert _call("dev_bar_pct", float("nan")) == "0"


def test_dev_bar_pct_scales_to_3_sigma():
    # |z| = 3 saturates to 100%
    assert _call("dev_bar_pct", 3.0) == "100.0"
    assert _call("dev_bar_pct", -3.0) == "100.0"
    # Beyond 3σ is still capped at 100
    assert _call("dev_bar_pct", 5.0) == "100.0"
    # Mid-range
    assert _call("dev_bar_pct", 1.5) == "50.0"
    assert _call("dev_bar_pct", -0.6) == "20.0"


def test_dev_bar_pct_handles_unparseable():
    assert _call("dev_bar_pct", "not a number") == "0"


def test_dev_sign_class_cases():
    assert _call("dev_sign_class", 0.5) == "dev-pos"
    assert _call("dev_sign_class", -0.5) == "dev-neg"
    assert _call("dev_sign_class", 0.0) == "dev-zero"
    assert _call("dev_sign_class", None) == "dev-zero"


def test_fmt_signed_z_formats_with_sigma():
    assert _call("fmt_signed_z", 1.234) == "+1.23σ"
    assert _call("fmt_signed_z", -0.50000) == "-0.50σ"
    assert _call("fmt_signed_z", 0.0) == "+0.00σ"
    assert _call("fmt_signed_z", None) == "—"


def test_data_source_dataclass_defaults():
    ds = DataSource(name="x", source="t", join_key="entity_id")
    assert ds.as_list is False
    assert ds.limit == 1
    assert ds.order_by is None


def test_template_loader_parses_as_list_with_default_limit_50():
    text = (
        "---\n"
        "data:\n"
        "  feature_deviation:\n"
        "    source: 'v_account_feature_deviation_topn'\n"
        "    join_key: 'entity_id'\n"
        "    as_list: true\n"
        "---\n"
        "<div>{{entity_id}}</div>\n"
    )
    front, body = _parse_frontmatter(text)
    assert front["data"]["feature_deviation"]["as_list"] is True
    assert "<div>" in body


def test_template_loader_keeps_limit_at_1_for_non_list_default():
    # Belt-and-braces: we don't accidentally bump non-list sources to 50.
    import os
    import tempfile

    from src.template import load_template
    fm = (
        "---\n"
        "data:\n"
        "  account:\n"
        "    source: 'v_account_profile_x'\n"
        "    join_key: 'entity_id'\n"
        "---\n"
        "<div>x</div>\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(fm)
        path = f.name
    try:
        tpl = load_template(path)
        assert len(tpl.data_sources) == 1
        ds = tpl.data_sources[0]
        assert ds.as_list is False
        assert ds.limit == 1
    finally:
        os.unlink(path)


def test_template_loader_list_source_uses_limit_50_default():
    import os
    import tempfile

    from src.template import load_template
    fm = (
        "---\n"
        "data:\n"
        "  feature_deviation:\n"
        "    source: 'v_account_feature_deviation_topn'\n"
        "    join_key: 'entity_id'\n"
        "    as_list: true\n"
        "---\n"
        "<div>x</div>\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(fm)
        path = f.name
    try:
        tpl = load_template(path)
        ds = tpl.data_sources[0]
        assert ds.as_list is True
        assert ds.limit == 50
    finally:
        os.unlink(path)


def test_template_loader_explicit_limit_overrides_default():
    import os
    import tempfile

    from src.template import load_template
    fm = (
        "---\n"
        "data:\n"
        "  feature_deviation:\n"
        "    source: 'v'\n"
        "    join_key: 'entity_id'\n"
        "    as_list: true\n"
        "    limit: 5\n"
        "---\n"
        "<div>x</div>\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(fm)
        path = f.name
    try:
        tpl = load_template(path)
        assert tpl.data_sources[0].limit == 5
    finally:
        os.unlink(path)
