import pytest
import solution

def test_discount_once():
    assert solution.cart_total([(10.0, 2)], 10) == 18.0

def test_qty():
    assert solution.cart_total([(2.5, 4)]) == 10.0

def test_empty():
    assert solution.cart_total([]) == 0

def test_invalid_negative():
    with pytest.raises(ValueError):
        solution.cart_total([(1, 1)], -5)

def test_zero_discount_cents():
    assert solution.cart_total([(1.25, 1)], 0) == 1.25

def test_invalid_over_100():
    with pytest.raises(ValueError):
        solution.cart_total([(1, 1)], 150)
