import pytest
import warnings
import importlib
import sys


def test_qdp_import_without_extension_warns():
    # Force fresh import
    sys.modules.pop("qumat.qdp", None)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        module = importlib.import_module("qumat.qdp")

        # Ensure fallback stubs exist
        assert hasattr(module, "QdpEngine")
        assert hasattr(module, "QuantumTensor")

        # Ensure warning was emitted
        assert any(
            "QDP module not available" in str(warn.message)
            and warn.category is ImportWarning
            for warn in w
        )


def test_qdp_stub_engine_raises_import_error():
    import qumat.qdp as qdp

    with pytest.raises(ImportError) as exc:
        qdp.QdpEngine()

    assert "qumat-qdp native extension" in str(exc.value)


def test_qdp_stub_tensor_raises_import_error():
    import qumat.qdp as qdp

    with pytest.raises(ImportError):
        qdp.QuantumTensor()


def test_qdp_all_exports():
    import qumat.qdp as qdp

    assert "QdpEngine" in qdp.__all__
    assert "QuantumTensor" in qdp.__all__


def test_make_stub_creates_class():
    # Force fresh import
    sys.modules.pop("qumat.qdp", None)
    module = importlib.import_module("qumat.qdp")

    # Directly test stub factory
    stub_class = module._make_stub("TestStub")

    assert stub_class.__name__ == "TestStub"

    with pytest.raises(ImportError):
        stub_class()