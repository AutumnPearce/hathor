import pytest
from hathor import Hathor

def test_hathor_imports():
    """Test that Hathor can be imported."""
    assert Hathor is not None

def test_hathor_initiation():
    hathor = Hathor()
    assert hathor is not None
