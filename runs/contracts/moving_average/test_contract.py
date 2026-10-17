import pytest
import solution

def test_basic():
    assert solution.moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]

def test_full():
    assert solution.moving_average([2, 4], 2) == [3.0]

def test_short():
    assert solution.moving_average([1], 3) == []

def test_zero_window():
    with pytest.raises(ValueError):
        solution.moving_average([1, 2], 0)

def test_negative_window():
    with pytest.raises(ValueError):
        solution.moving_average([1, 2], -1)

def test_empty():
    assert solution.moving_average([], 1) == []
