import pytest

def test_hathor_imports():
    """Test that Hathor can be imported."""
    from hathor import Hathor
    assert Hathor is not None