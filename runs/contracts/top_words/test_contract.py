import pytest
import solution

def test_basic():
    assert solution.top_words('a b a c a b', 2) == [('a', 3), ('b', 2)]

def test_case():
    assert solution.top_words('Hi hi HI', 1) == [('hi', 3)]

def test_ties():
    assert solution.top_words('b a c', 2) == [('a', 1), ('b', 1)]

def test_negative_k():
    with pytest.raises(ValueError):
        solution.top_words('a', -1)

def test_zero_k():
    assert solution.top_words('a b', 0) == []
