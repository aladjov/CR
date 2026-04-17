from __future__ import annotations


class TestNamespaceFacade:
    def test_cr_importable_via_runtime_package(self):
        from customer_retention.runtime import cr
        assert hasattr(cr, "register")
        assert hasattr(cr, "log")
        assert hasattr(cr, "log_table")
        assert hasattr(cr, "in_notebook")
        assert hasattr(cr, "registry")

    def test_runtime_not_re_exported_from_top_level_package(self):
        """Plan § 3 intro: 'customer_retention/__init__.py does NOT
        re-export cr — users write `from customer_retention.runtime import cr`'."""
        import customer_retention as pkg
        assert not hasattr(pkg, "cr")

    def test_registry_is_shared_singleton(self):
        from customer_retention.runtime import cr
        from customer_retention.runtime.registry import registry as direct
        assert cr.registry is direct
