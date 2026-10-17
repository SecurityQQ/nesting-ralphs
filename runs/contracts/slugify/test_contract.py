import pytest
import solution

def test_basic():
    assert solution.slugify('Hello World') == 'hello-world'

def test_punctuation():
    assert solution.slugify('Rock & Roll!') == 'rock-roll'

def test_leading():
    assert solution.slugify('  --hi there') == 'hi-there'

def test_truncate():
    assert solution.slugify('abc def', max_len=4) == 'abc'

def test_trailing():
    assert solution.slugify('hi!') == 'hi'

def test_exact_len():
    assert solution.slugify('abcdef', max_len=6) == 'abcdef'
