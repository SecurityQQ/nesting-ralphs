import pytest
import solution

def test_minutes():
    assert solution.parse_duration('90m') == 5400

def test_hours():
    assert solution.parse_duration('2h') == 7200

def test_combined():
    assert solution.parse_duration('1h30m15s') == 5415

def test_empty():
    with pytest.raises(ValueError):
        solution.parse_duration('')

def test_garbage():
    with pytest.raises(ValueError):
        solution.parse_duration('abc')

def test_case_whitespace():
    assert solution.parse_duration('  5S ') == 5
