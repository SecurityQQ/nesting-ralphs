"""Fixture tasks.

Each task is a small buggy Python module. Editable regions ("sites") have a
finite set of variants: index 0 = original (buggy), index 1 = correct,
index >=2 = plausible-but-wrong. The executor edits sites; the independent
verifier runs the frozen contract tests with pytest on the real file.

`visible` criteria are what the executor observes after every step.
The verifier checks ALL mandatory criteria, including hidden ones.
`hints` map a criterion to the site an agent would inspect after reading the
failing test message (used only by the simulated executor policy).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    name: str
    goal: str
    template: str
    sites: tuple  # tuple[tuple[str, ...], ...]
    tests: dict  # criterion_id -> test body (python source, uses `solution`)
    visible: tuple
    hints: dict = field(default_factory=dict)

    def render(self, variant):
        src = self.template
        for i, v in enumerate(variant):
            src = src.replace(f"#@S{i}@", self.sites[i][v])
        return src

    @property
    def root(self):
        return tuple(0 for _ in self.sites)

    @property
    def criteria(self):
        return tuple(self.tests)


T = []

T.append(Task(
    name="cart_discount",
    goal="Fix cart_total: discount is a percentage in [0, 100] applied exactly once to the subtotal; result rounded to cents.",
    template='''def cart_total(items, discount_pct=0):
    """items: list of (price, qty). discount_pct in [0, 100], applied once."""
#@S0@
    subtotal = 0
    for price, qty in items:
#@S1@
#@S2@
''',
    sites=(
        ("    pass",
         "    if not 0 <= discount_pct <= 100:\n        raise ValueError('discount_pct out of range')",
         "    if discount_pct < 0:\n        raise ValueError('negative discount')"),
        ("        subtotal += price * qty * (1 - discount_pct / 100)",
         "        subtotal += price * qty",
         "        subtotal += price"),
        ("    return int(subtotal * (1 - discount_pct / 100))",
         "    return round(subtotal * (1 - discount_pct / 100), 2)",
         "    return round(subtotal - discount_pct, 2)"),
    ),
    tests={
        "discount_once": "assert solution.cart_total([(10.0, 2)], 10) == 18.0",
        "qty": "assert solution.cart_total([(2.5, 4)]) == 10.0",
        "empty": "assert solution.cart_total([]) == 0",
        "invalid_negative": "with pytest.raises(ValueError):\n        solution.cart_total([(1, 1)], -5)",
        "zero_discount_cents": "assert solution.cart_total([(1.25, 1)], 0) == 1.25",
        "invalid_over_100": "with pytest.raises(ValueError):\n        solution.cart_total([(1, 1)], 150)",
    },
    visible=("discount_once", "qty", "empty", "invalid_negative"),
    hints={"discount_once": 1, "qty": 1, "invalid_negative": 0, "empty": 2},
))

T.append(Task(
    name="parse_duration",
    goal="Fix parse_duration: '1h30m15s' -> seconds; units h/m/s; empty or malformed input raises ValueError.",
    template='''import re


def parse_duration(text):
    """Parse '1h30m15s' style durations into seconds."""
    units = {
#@S0@
    }
    parts = re.findall(r"(\\d+)([a-z])", text.strip().lower())
#@S1@
    total = 0
    for num, unit in parts:
        if unit not in units:
            raise ValueError(f"unknown unit {unit}")
#@S2@
    return total
''',
    sites=(
        ('        "h": 3600, "m": 3600, "s": 1,',
         '        "h": 3600, "m": 60, "s": 1,',
         '        "h": 360, "m": 60, "s": 1,'),
        ("    pass",
         "    if not parts or ''.join(n + u for n, u in parts) != text.strip().lower():\n        raise ValueError('bad duration')",
         "    if not text:\n        raise ValueError('empty')"),
        ("        total = int(num) * units[unit]",
         "        total += int(num) * units[unit]",
         "        total += int(num)"),
    ),
    tests={
        "minutes": "assert solution.parse_duration('90m') == 5400",
        "hours": "assert solution.parse_duration('2h') == 7200",
        "combined": "assert solution.parse_duration('1h30m15s') == 5415",
        "empty": "with pytest.raises(ValueError):\n        solution.parse_duration('')",
        "garbage": "with pytest.raises(ValueError):\n        solution.parse_duration('abc')",
        "case_whitespace": "assert solution.parse_duration('  5S ') == 5",
    },
    visible=("minutes", "hours", "combined", "empty"),
    hints={"minutes": 0, "hours": 0, "combined": 2, "empty": 1},
))

T.append(Task(
    name="clamp_range",
    goal="Fix clamp: clamp x into [lo, hi]; lo > hi raises ValueError; lo == hi is allowed.",
    template='''def clamp(x, lo, hi):
    """Clamp x into [lo, hi]."""
#@S0@
#@S1@
''',
    sites=(
        ("    pass",
         "    if lo > hi:\n        raise ValueError('lo > hi')",
         "    if lo >= hi:\n        raise ValueError('lo >= hi')"),
        ("    return max(x, min(lo, hi))",
         "    return max(lo, min(x, hi))",
         "    return min(lo, max(x, hi))"),
    ),
    tests={
        "inside": "assert solution.clamp(5, 0, 10) == 5",
        "below": "assert solution.clamp(-3, 0, 10) == 0",
        "above": "assert solution.clamp(42, 0, 10) == 10",
        "invalid": "with pytest.raises(ValueError):\n        solution.clamp(1, 5, 0)",
        "degenerate": "assert solution.clamp(7, 3, 3) == 3",
        "float": "assert solution.clamp(0.5, 0.0, 1.0) == 0.5",
    },
    visible=("inside", "below", "above", "invalid"),
    hints={"inside": 1, "below": 1, "above": 1, "invalid": 0},
))

T.append(Task(
    name="slugify",
    goal="Fix slugify: lowercase ASCII words joined by '-', no leading/trailing '-', truncated to max_len without a trailing '-'.",
    template='''import re


def slugify(title, max_len=40):
    """URL slug."""
#@S0@
    s = re.sub(r"[^a-z0-9]+", "-", s)
#@S1@
#@S2@
    return s
''',
    sites=(
        ("    s = title",
         "    s = title.lower()",
         "    s = title.lower().replace(' ', '')"),
        ("    s = s.strip()",
         "    s = s.strip('-')",
         "    s = s.lstrip('-')"),
        ("    s = s[:max_len]",
         "    s = s[:max_len].rstrip('-')",
         "    s = s[:max_len - 1]"),
    ),
    tests={
        "basic": "assert solution.slugify('Hello World') == 'hello-world'",
        "punctuation": "assert solution.slugify('Rock & Roll!') == 'rock-roll'",
        "leading": "assert solution.slugify('  --hi there') == 'hi-there'",
        "truncate": "assert solution.slugify('abc def', max_len=4) == 'abc'",
        "trailing": "assert solution.slugify('hi!') == 'hi'",
        "exact_len": "assert solution.slugify('abcdef', max_len=6) == 'abcdef'",
    },
    visible=("basic", "punctuation", "leading", "truncate"),
    hints={"basic": 0, "punctuation": 1, "leading": 1, "truncate": 2},
))

T.append(Task(
    name="moving_average",
    goal="Fix moving_average: mean over every full window; window < 1 raises ValueError; [] if fewer items than window.",
    template='''def moving_average(xs, window):
    """Simple moving average over full windows."""
#@S0@
    out = []
#@S1@
        chunk = xs[i:i + window]
#@S2@
    return out
''',
    sites=(
        ("    pass",
         "    if window < 1:\n        raise ValueError('window must be >= 1')",
         "    if window == 0:\n        raise ValueError('window must be nonzero')"),
        ("    for i in range(len(xs) - window):",
         "    for i in range(len(xs) - window + 1):",
         "    for i in range(len(xs)):"),
        ("        out.append(sum(chunk) // window)",
         "        out.append(sum(chunk) / window)",
         "        out.append(sum(chunk) / len(xs))"),
    ),
    tests={
        "basic": "assert solution.moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]",
        "full": "assert solution.moving_average([2, 4], 2) == [3.0]",
        "short": "assert solution.moving_average([1], 3) == []",
        "zero_window": "with pytest.raises(ValueError):\n        solution.moving_average([1, 2], 0)",
        "negative_window": "with pytest.raises(ValueError):\n        solution.moving_average([1, 2], -1)",
        "empty": "assert solution.moving_average([], 1) == []",
    },
    visible=("basic", "full", "short", "zero_window"),
    hints={"basic": 2, "full": 1, "short": 1, "zero_window": 0},
))

T.append(Task(
    name="roman_to_int",
    goal="Fix roman_to_int: standard subtractive notation; empty or invalid characters raise ValueError.",
    template='''def roman_to_int(s):
    """Convert a Roman numeral to int."""
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
#@S0@
    total = 0
    for i, ch in enumerate(s):
        v = vals[ch]
#@S1@
    return total
''',
    sites=(
        ("    pass",
         "    if not s or any(c not in vals for c in s):\n        raise ValueError('invalid numeral')",
         "    if not s:\n        raise ValueError('empty')"),
        ("        total += v",
         "        if i + 1 < len(s) and vals[s[i + 1]] > v:\n            total -= v\n        else:\n            total += v",
         "        if i + 1 < len(s) and vals[s[i + 1]] > v:\n            total -= 2 * v\n        else:\n            total += v"),
    ),
    tests={
        "simple": "assert solution.roman_to_int('III') == 3",
        "subtractive": "assert solution.roman_to_int('IV') == 4",
        "complex": "assert solution.roman_to_int('MCMXCIV') == 1994",
        "empty": "with pytest.raises(ValueError):\n        solution.roman_to_int('')",
        "invalid_chars": "with pytest.raises(ValueError):\n        solution.roman_to_int('XQ')",
        "max_value": "assert solution.roman_to_int('MMMCMXCIX') == 3999",
    },
    visible=("simple", "subtractive", "complex", "empty"),
    hints={"subtractive": 1, "complex": 1, "empty": 0},
))

T.append(Task(
    name="median",
    goal="Fix median: median of a non-empty list without mutating the input; empty list raises ValueError.",
    template='''def median(xs):
    """Median of a non-empty list of numbers."""
#@S0@
#@S1@
    n = len(s)
    mid = n // 2
#@S2@
''',
    sites=(
        ("    pass",
         "    if not xs:\n        raise ValueError('empty')",
         "    if xs is None:\n        raise ValueError('none')"),
        ("    xs.sort(); s = xs",
         "    s = sorted(xs)",
         "    s = list(xs)"),
        ("    return s[mid]",
         "    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2",
         "    return (s[mid - 1] + s[mid]) / 2"),
    ),
    tests={
        "odd": "assert solution.median([3, 1, 2]) == 2",
        "even": "assert solution.median([4, 1, 3, 2]) == 2.5",
        "empty": "with pytest.raises(ValueError):\n        solution.median([])",
        "no_mutation": "xs = [3, 1, 2]\n    solution.median(xs)\n    assert xs == [3, 1, 2]",
        "negatives": "assert solution.median([-5, -1, -3]) == -3",
    },
    visible=("odd", "even", "empty"),
    hints={"odd": 1, "even": 2, "empty": 0},
))

T.append(Task(
    name="top_words",
    goal="Fix top_words: k most frequent lowercase words as (word, count), ties alphabetical; k < 0 raises ValueError; k == 0 returns [].",
    template='''import re
from collections import Counter


def top_words(text, k):
    """Top-k word frequencies."""
#@S0@
    counts = Counter(re.findall(r"[a-z']+", text.lower()))
#@S1@
    return ranked[:k]
''',
    sites=(
        ("    pass",
         "    if k < 0:\n        raise ValueError('k must be >= 0')",
         "    if k <= 0:\n        raise ValueError('k must be > 0')"),
        ("    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)",
         "    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))",
         "    ranked = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)"),
    ),
    tests={
        "basic": "assert solution.top_words('a b a c a b', 2) == [('a', 3), ('b', 2)]",
        "case": "assert solution.top_words('Hi hi HI', 1) == [('hi', 3)]",
        "ties": "assert solution.top_words('b a c', 2) == [('a', 1), ('b', 1)]",
        "negative_k": "with pytest.raises(ValueError):\n        solution.top_words('a', -1)",
        "zero_k": "assert solution.top_words('a b', 0) == []",
    },
    visible=("basic", "case", "ties", "negative_k"),
    hints={"ties": 1, "basic": 1, "negative_k": 0},
))

TASKS = {t.name: t for t in T}
# Split by task family BEFORE any state/prefix extraction.
TRAIN = ["cart_discount", "parse_duration", "clamp_range", "slugify"]
VALID = ["moving_average"]  # epoch/optimizer selection only
TEST = ["roman_to_int", "median", "top_words"]  # sealed: never used for training or selection
