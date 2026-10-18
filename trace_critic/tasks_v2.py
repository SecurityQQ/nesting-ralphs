"""Free-form tasks for the Nemotron executor.

Each task: goal (the one-line prompt), buggy source, hidden tests (the independent verifier,
never shown to the agent or to the DoD generator) and a reference solution used ONLY to
sanity-check that the hidden tests are correct and to measure generated-DoD quality.
"""

from dataclasses import dataclass

from .tasks import TASKS as V1


@dataclass(frozen=True)
class FTask:
    name: str
    func: str
    goal: str
    buggy: str
    reference: str
    hidden: dict  # criterion -> test body (dedented python, uses `solution`, `pytest`)


def _from_v1(t, func):
    hidden = {k: v.replace("\n    ", "\n") for k, v in t.tests.items()}
    return FTask(t.name, func, t.goal, t.render(t.root), t.render(tuple(1 for _ in t.sites)), hidden)


F = [
    _from_v1(V1["cart_discount"], "cart_total"), _from_v1(V1["parse_duration"], "parse_duration"),
    _from_v1(V1["clamp_range"], "clamp"), _from_v1(V1["slugify"], "slugify"),
    _from_v1(V1["moving_average"], "moving_average"), _from_v1(V1["roman_to_int"], "roman_to_int"),
    _from_v1(V1["median"], "median"), _from_v1(V1["top_words"], "top_words"),
]

F.append(FTask("merge_intervals", "merge_intervals",
    "Fix merge_intervals(intervals): merge overlapping [start, end] pairs; touching intervals like [1,2],[2,3] merge; "
    "input may be unsorted and must not be mutated; return a new sorted list of [start, end] lists; [] -> [].",
    '''def merge_intervals(intervals):
    out = []
    for s, e in intervals:
        if out and s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
''',
    '''def merge_intervals(intervals):
    out = []
    for s, e in sorted([list(i) for i in intervals]):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
''',
    {"basic": "assert solution.merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]",
     "unsorted": "assert solution.merge_intervals([[8, 10], [1, 3], [2, 6]]) == [[1, 6], [8, 10]]",
     "touching": "assert solution.merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]",
     "no_mutation": "x = [[3, 4], [1, 2]]\nsolution.merge_intervals(x)\nassert x == [[3, 4], [1, 2]]",
     "empty": "assert solution.merge_intervals([]) == []",
     "contained": "assert solution.merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]"}))

F.append(FTask("split_bill", "split_bill",
    "Fix split_bill(total_cents, n): split an integer amount of cents among n people so shares differ by at most "
    "1 cent and sum exactly to total_cents, larger shares first; n <= 0 raises ValueError.",
    '''def split_bill(total_cents, n):
    share = total_cents // n
    return [share] * n
''',
    '''def split_bill(total_cents, n):
    if n <= 0:
        raise ValueError("n must be positive")
    q, r = divmod(total_cents, n)
    return [q + 1] * r + [q] * (n - r)
''',
    {"even": "assert solution.split_bill(900, 3) == [300, 300, 300]",
     "remainder": "assert solution.split_bill(1000, 3) == [334, 333, 333]",
     "sum_exact": "assert sum(solution.split_bill(10001, 7)) == 10001",
     "zero_people": "with pytest.raises(ValueError):\n    solution.split_bill(100, 0)",
     "negative_people": "with pytest.raises(ValueError):\n    solution.split_bill(100, -2)",
     "max_diff": "s = solution.split_bill(5, 4)\nassert max(s) - min(s) <= 1 and s == sorted(s, reverse=True)"}))

F.append(FTask("version_compare", "compare_versions",
    "Fix compare_versions(a, b): compare dotted versions numerically ('1.10' > '1.9'); missing parts count as 0 "
    "('1.0' == '1'); a pre-release suffix like '1.0.0-rc1' is lower than '1.0.0'; return -1, 0 or 1.",
    '''def compare_versions(a, b):
    if a == b:
        return 0
    return 1 if a > b else -1
''',
    '''def compare_versions(a, b):
    def parse(v):
        core, _, pre = v.partition("-")
        nums = [int(x) for x in core.split(".")]
        return nums, pre
    (na, pa), (nb, pb) = parse(a), parse(b)
    n = max(len(na), len(nb))
    na += [0] * (n - len(na))
    nb += [0] * (n - len(nb))
    if na != nb:
        return 1 if na > nb else -1
    if pa == pb:
        return 0
    if not pa:
        return 1
    if not pb:
        return -1
    return 1 if pa > pb else -1
''',
    {"numeric": "assert solution.compare_versions('1.10', '1.9') == 1",
     "equal": "assert solution.compare_versions('2.0.1', '2.0.1') == 0",
     "missing_parts": "assert solution.compare_versions('1.0', '1') == 0",
     "less": "assert solution.compare_versions('0.9.9', '1.0') == -1",
     "prerelease": "assert solution.compare_versions('1.0.0-rc1', '1.0.0') == -1",
     "prerelease_rev": "assert solution.compare_versions('1.0.0', '1.0.0-rc1') == 1"}))

F.append(FTask("chunk_list", "chunk",
    "Fix chunk(items, size): split any iterable into consecutive lists of length size, the last one may be "
    "shorter; size <= 0 raises ValueError; empty input returns [].",
    '''def chunk(items, size):
    return [items[i:i + size] for i in range(0, len(items) - size, size)]
''',
    '''def chunk(items, size):
    if size <= 0:
        raise ValueError("size must be positive")
    items = list(items)
    return [items[i:i + size] for i in range(0, len(items), size)]
''',
    {"exact": "assert solution.chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]",
     "remainder": "assert solution.chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
     "empty": "assert solution.chunk([], 3) == []",
     "zero": "with pytest.raises(ValueError):\n    solution.chunk([1], 0)",
     "generator": "assert solution.chunk((x for x in range(3)), 2) == [[0, 1], [2]]",
     "tuple_to_lists": "assert solution.chunk((1, 2, 3), 2) == [[1, 2], [3]]"}))

F.append(FTask("luhn", "luhn_valid",
    "Fix luhn_valid(number): return True if the digit string passes the Luhn checksum; spaces are allowed and "
    "ignored; any other non-digit character, or fewer than 2 digits, returns False.",
    '''def luhn_valid(number):
    digits = [int(c) for c in number]
    total = 0
    for i, d in enumerate(digits):
        if i % 2 == 1:
            d *= 2
        total += d
    return total % 10 == 0
''',
    '''def luhn_valid(number):
    s = number.replace(" ", "")
    if len(s) < 2 or not s.isdigit():
        return False
    total = 0
    for i, c in enumerate(reversed(s)):
        d = int(c)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
''',
    {"valid": "assert solution.luhn_valid('4539 3195 0343 6467') is True",
     "invalid": "assert solution.luhn_valid('8273 1232 7352 0569') is False",
     "odd_length": "assert solution.luhn_valid('059') is True",
     "letters": "assert solution.luhn_valid('055a 444 285') is False",
     "single_digit": "assert solution.luhn_valid('0') is False",
     "double_over_9": "assert solution.luhn_valid('91') is True"}))

F.append(FTask("dedupe", "dedupe",
    "Fix dedupe(items, key=None): remove duplicates keeping the first occurrence and original order; optional key "
    "function decides identity; must also work for unhashable items such as lists.",
    '''def dedupe(items, key=None):
    return list(set(items))
''',
    '''def dedupe(items, key=None):
    out, seen_h, seen_u = [], set(), []
    for it in items:
        k = key(it) if key else it
        try:
            if k in seen_h:
                continue
            seen_h.add(k)
        except TypeError:
            if k in seen_u:
                continue
            seen_u.append(k)
        out.append(it)
    return out
''',
    {"order": "assert solution.dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]",
     "strings": "assert solution.dedupe(['b', 'a', 'b']) == ['b', 'a']",
     "key": "assert solution.dedupe(['Hi', 'hi', 'HO'], key=str.lower) == ['Hi', 'HO']",
     "unhashable": "assert solution.dedupe([[1], [2], [1]]) == [[1], [2]]",
     "empty": "assert solution.dedupe([]) == []",
     "key_first_wins": "assert solution.dedupe([(1, 'a'), (1, 'b')], key=lambda t: t[0]) == [(1, 'a')]"}))

F.append(FTask("to_snake", "to_snake",
    "Fix to_snake(name): convert CamelCase/camelCase identifiers to snake_case; acronyms stay together "
    "('HTTPServerError' -> 'http_server_error', 'getHTTP' -> 'get_http'); existing snake_case is unchanged; "
    "digits stick to the preceding word ('version2Update' -> 'version2_update').",
    '''import re


def to_snake(name):
    return re.sub(r"([A-Z])", r"_\\1", name).lower()
''',
    '''import re


def to_snake(name):
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\\1_\\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", s)
    return s.lower()
''',
    {"camel": "assert solution.to_snake('camelCase') == 'camel_case'",
     "pascal": "assert solution.to_snake('PascalCase') == 'pascal_case'",
     "acronym_start": "assert solution.to_snake('HTTPServerError') == 'http_server_error'",
     "acronym_end": "assert solution.to_snake('getHTTP') == 'get_http'",
     "snake_unchanged": "assert solution.to_snake('already_snake') == 'already_snake'",
     "digits": "assert solution.to_snake('version2Update') == 'version2_update'"}))

F.append(FTask("business_days", "business_days",
    "Fix business_days(start, end): number of weekdays (Mon-Fri) from start (inclusive) to end (exclusive) for "
    "datetime.date values; if end < start return the negative of business_days(end, start); same day -> 0.",
    '''import datetime


def business_days(start, end):
    days = (end - start).days
    return days - (days // 7) * 2
''',
    '''import datetime


def business_days(start, end):
    if end < start:
        return -business_days(end, start)
    n, d = 0, start
    while d < end:
        if d.weekday() < 5:
            n += 1
        d += datetime.timedelta(days=1)
    return n
''',
    {"week": "import datetime as dt\nassert solution.business_days(dt.date(2026, 10, 12), dt.date(2026, 10, 19)) == 5",
     "weekend_start": "import datetime as dt\nassert solution.business_days(dt.date(2026, 10, 17), dt.date(2026, 10, 20)) == 1",
     "same_day": "import datetime as dt\nassert solution.business_days(dt.date(2026, 10, 14), dt.date(2026, 10, 14)) == 0",
     "reverse": "import datetime as dt\nassert solution.business_days(dt.date(2026, 10, 19), dt.date(2026, 10, 12)) == -5",
     "midweek": "import datetime as dt\nassert solution.business_days(dt.date(2026, 10, 15), dt.date(2026, 10, 21)) == 4",
     "long": "import datetime as dt\nassert solution.business_days(dt.date(2026, 1, 1), dt.date(2026, 12, 31)) == 260"}))

F.append(FTask("flatten", "flatten",
    "Fix flatten(nested, depth=None): flatten nested lists and tuples into one list; strings and bytes are atoms "
    "and are never split; depth limits how many levels are removed (depth=1 removes one level); None = fully.",
    '''def flatten(nested, depth=None):
    out = []
    for x in nested:
        if hasattr(x, "__iter__"):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out
''',
    '''def flatten(nested, depth=None):
    out = []
    for x in nested:
        if isinstance(x, (list, tuple)) and (depth is None or depth > 0):
            out.extend(flatten(x, None if depth is None else depth - 1))
        else:
            out.append(x)
    return out
''',
    {"deep": "assert solution.flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]",
     "tuples": "assert solution.flatten([(1, 2), [3]]) == [1, 2, 3]",
     "strings": "assert solution.flatten(['ab', ['cd']]) == ['ab', 'cd']",
     "depth1": "assert solution.flatten([1, [2, [3]]], depth=1) == [1, 2, [3]]",
     "depth0": "assert solution.flatten([1, [2]], depth=0) == [1, [2]]",
     "bytes": "assert solution.flatten([b'xy', [b'z']]) == [b'xy', b'z']"}))

F.append(FTask("percentile", "percentile",
    "Fix percentile(xs, p): p-th percentile with linear interpolation between closest ranks (numpy default); "
    "p must be in [0, 100] else ValueError; empty xs raises ValueError; do not mutate xs.",
    '''def percentile(xs, p):
    xs.sort()
    k = int(len(xs) * p / 100)
    return xs[k]
''',
    '''def percentile(xs, p):
    if not xs:
        raise ValueError("empty")
    if not 0 <= p <= 100:
        raise ValueError("p out of range")
    s = sorted(xs)
    pos = (len(s) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)
''',
    {"median": "assert solution.percentile([1, 2, 3, 4], 50) == 2.5",
     "max": "assert solution.percentile([5, 1, 3], 100) == 5",
     "min": "assert solution.percentile([5, 1, 3], 0) == 1",
     "interp": "assert abs(solution.percentile([10, 20, 30, 40], 25) - 17.5) < 1e-9",
     "range": "with pytest.raises(ValueError):\n    solution.percentile([1, 2], 101)",
     "no_mutation": "x = [3, 1, 2]\nsolution.percentile(x, 50)\nassert x == [3, 1, 2]"}))

F.append(FTask("word_wrap", "wrap",
    "Fix wrap(text, width): greedy word wrap returning a list of lines; collapse runs of whitespace; no line has "
    "leading/trailing spaces; a word longer than width goes on its own line unbroken; empty text -> [].",
    '''def wrap(text, width):
    lines, cur = [], ""
    for w in text.split(" "):
        if len(cur) + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur += " " + w
    lines.append(cur)
    return lines
''',
    '''def wrap(text, width):
    lines, cur = [], ""
    for w in text.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
''',
    {"basic": "assert solution.wrap('the quick brown fox', 10) == ['the quick', 'brown fox']",
     "exact_fit": "assert solution.wrap('ab cd', 5) == ['ab cd']",
     "long_word": "assert solution.wrap('a supercalifragilistic b', 6) == ['a', 'supercalifragilistic', 'b']",
     "whitespace": "assert solution.wrap('  a   b  ', 10) == ['a b']",
     "empty": "assert solution.wrap('', 5) == []",
     "no_trailing": "assert all(l == l.strip() for l in solution.wrap('one two three four five', 7))"}))

F.append(FTask("csv_line", "parse_csv_line",
    "Fix parse_csv_line(line): split one CSV line on commas; fields may be double-quoted and then can contain "
    "commas; a doubled quote \"\" inside quotes is a literal quote; whitespace is preserved; '' -> [''].",
    '''def parse_csv_line(line):
    return [f.strip('"') for f in line.split(",")]
''',
    '''def parse_csv_line(line):
    out, cur, q, i = [], "", False, 0
    while i < len(line):
        c = line[i]
        if q:
            if c == '"' and i + 1 < len(line) and line[i + 1] == '"':
                cur += '"'
                i += 1
            elif c == '"':
                q = False
            else:
                cur += c
        elif c == '"':
            q = True
        elif c == ",":
            out.append(cur)
            cur = ""
        else:
            cur += c
        i += 1
    out.append(cur)
    return out
''',
    {"plain": "assert solution.parse_csv_line('a,b,c') == ['a', 'b', 'c']",
     "quoted_comma": "assert solution.parse_csv_line('\"x,y\",z') == ['x,y', 'z']",
     "escaped_quote": "assert solution.parse_csv_line('\"say \"\"hi\"\"\",1') == ['say \"hi\"', '1']",
     "empty_fields": "assert solution.parse_csv_line('a,,b,') == ['a', '', 'b', '']",
     "whitespace": "assert solution.parse_csv_line(' a , b') == [' a ', ' b']",
     "empty_line": "assert solution.parse_csv_line('') == ['']"}))

FTASKS = {t.name: t for t in F}
# Split re-drawn after a 60-episode pilot, stratified by pilot difficulty so the sealed test set is not at
# ceiling (pilot outcomes are not used for anything else). Disclosed in README.
TRAIN2 = ["cart_discount", "parse_duration", "clamp_range", "moving_average", "split_bill", "chunk_list",
          "luhn", "dedupe", "business_days", "top_words", "to_snake", "percentile"]
VALID2 = ["slugify", "merge_intervals"]
TEST2 = ["roman_to_int", "version_compare", "flatten", "word_wrap", "median", "csv_line"]
assert sorted(TRAIN2 + VALID2 + TEST2) == sorted(FTASKS)
