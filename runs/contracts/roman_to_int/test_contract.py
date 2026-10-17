import pytest
import solution

def test_simple():
    assert solution.roman_to_int('III') == 3

def test_subtractive():
    assert solution.roman_to_int('IV') == 4

def test_complex():
    assert solution.roman_to_int('MCMXCIV') == 1994

def test_empty():
    with pytest.raises(ValueError):
        solution.roman_to_int('')

def test_invalid_chars():
    with pytest.raises(ValueError):
        solution.roman_to_int('XQ')

def test_max_value():
    assert solution.roman_to_int('MMMCMXCIX') == 3999
