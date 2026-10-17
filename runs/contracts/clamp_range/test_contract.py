import pytest
import solution

def test_inside():
    assert solution.clamp(5, 0, 10) == 5

def test_below():
    assert solution.clamp(-3, 0, 10) == 0

def test_above():
    assert solution.clamp(42, 0, 10) == 10

def test_invalid():
    with pytest.raises(ValueError):
        solution.clamp(1, 5, 0)

def test_degenerate():
    assert solution.clamp(7, 3, 3) == 3

def test_float():
    assert solution.clamp(0.5, 0.0, 1.0) == 0.5
