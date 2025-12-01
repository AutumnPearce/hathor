import pytest
from hathor import Hathor

def test_hathor_imports():
    """Test that Hathor can be imported."""
    assert Hathor is not None