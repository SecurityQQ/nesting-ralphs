import pytest
import solution

def test_odd():
    assert solution.median([3, 1, 2]) == 2

def test_even():
    assert solution.median([4, 1, 3, 2]) == 2.5

def test_empty():
    with pytest.raises(ValueError):
        solution.median([])

def test_no_mutation():
    xs = [3, 1, 2]
    solution.median(xs)
    assert xs == [3, 1, 2]

def test_negatives():
    assert solution.median([-5, -1, -3]) == -3
