# ruff: noqa: E731
# mypy: disable-error-code="misc,assignment,arg-type,no-any-return"
"""Evaluation suite v2: harder parametric tasks, 12 categories, statistics.

Every task is generated from a seed with disjoint train/validation/test pools
for any finite item list. Scoring is deterministic. ``compare`` gives paired
item-level significance (exact sign test) so a "better" model must win on the
same items, not only on a noisy mean. Nothing here calls a hosted model.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import time
from typing import Any, Callable

from .capability_eval import (
    ANSWER_SUFFIX,
    HE_ANSWER_SUFFIX,
    HE_CAPITALS,
    Task,
    _extract_code,
    _num_check,
    _pool,
    parse_tool_calls,
    run_python_tests,
)

SUITE_VERSION = 2
CATEGORIES = (
    "math", "coding", "reasoning", "hebrew", "english", "instruction_following", "tool_use",
    "long_context", "multi_turn", "factual", "general_assistance", "agentic",
)
SPLIT_SEEDS = {"train": 70_000_000, "validation": 71_000_000, "test": 72_000_000}
# Provisional frontier-class profile on THIS suite; not a measured GPT-5.6 Luna score.
TARGET_THRESHOLDS = {c: 0.90 for c in CATEGORIES}

# ---------------------------------------------------------------- math

def _math(rng: random.Random, i: int, split: str) -> Task:
    k = i % 8
    if k == 0:
        price, pct, qty, tax = rng.randint(20, 200), rng.choice([10, 15, 20, 25, 30]), rng.randint(2, 9), rng.choice([5, 10, 20])
        q = (f"An item costs ${price}. It is discounted by {pct}%, then a {tax}% tax is added to the discounted price. "
             f"What is the total for {qty} items, in cents?")
        ans = round(price * 100 * (100 - pct) * (100 + tax) * qty / 10000)
    elif k == 1:
        a, b = rng.randint(-9, 9), rng.randint(-9, 9)
        c1, d1, c2, d2 = rng.randint(1, 7), rng.randint(1, 7), rng.randint(1, 7), rng.randint(-7, -1)
        q = f"Solve the system: {c1}x + {d1}y = {c1 * a + d1 * b} and {c2}x + ({d2})y = {c2 * a + d2 * b}. What is x + 2y?"
        ans = a + 2 * b
    elif k == 2:
        n, r = rng.randint(6, 14), rng.randint(2, 5)
        q = f"How many ways are there to choose a committee of {r} people from {n} people?"
        ans = math.comb(n, r)
    elif k == 3:
        base, exp, mod = rng.randint(2, 30), rng.randint(20, 200), rng.choice([7, 11, 13, 17, 19, 23, 29])
        q = f"What is {base}^{exp} mod {mod}?"
        ans = pow(base, exp, mod)
    elif k == 4:
        a0, d, n = rng.randint(-20, 20), rng.randint(2, 9), rng.randint(15, 60)
        q = f"An arithmetic sequence starts at {a0} with common difference {d}. What is the sum of its first {n} terms?"
        ans = n * (2 * a0 + (n - 1) * d) // 2
    elif k == 5:
        x, y = rng.randint(12, 90), rng.randint(12, 90)
        q = f"What is the least common multiple of {x} and {y} minus their greatest common divisor?"
        ans = x * y // math.gcd(x, y) - math.gcd(x, y)
    elif k == 6:
        w, h, cut = rng.randint(8, 30), rng.randint(8, 30), rng.randint(1, 4)
        q = (f"A rectangle is {w} cm by {h} cm. A square of side {cut} cm is cut from each of its four corners and the "
             f"sides are folded up into an open box. What is the box volume in cubic cm?")
        ans = (w - 2 * cut) * (h - 2 * cut) * cut
    else:
        r1, r2 = rng.randint(2, 9), rng.randint(2, 9)
        q = (f"Pipe A alone fills a tank in {r1 * 2} hours and pipe B alone fills it in {r2 * 2} hours. "
             f"Working together, how many minutes do they need to fill the tank? Round down to an integer.")
        ans = int(60 * (r1 * 2) * (r2 * 2) / (r1 * 2 + r2 * 2))
    return Task("math", f"m{i}", [{"role": "user", "content": q + ANSWER_SUFFIX}], _num_check(ans), 768, reference=str(ans))


# ---------------------------------------------------------------- coding

def _coding(rng: random.Random, i: int, split: str) -> Task:
    """Parametric coding templates: every prompt parameter changes the required behavior."""
    k = rng.randrange(12)
    cases: list[tuple[Any, ...]]
    if k == 0:
        m = rng.randint(2, 9)
        name, desc = f"sum_div{m}_squares", f"takes a list of integers and returns the sum of squares of the elements divisible by {m}"
        cases = [([rng.randint(-30, 30) for _ in range(rng.randint(0, 9))],) for _ in range(6)]
        exp = [sum(x * x for x in c[0] if x % m == 0) for c in cases]
    elif k == 1:
        kk, larger = rng.randint(1, 3), rng.random() < 0.5
        name = "top_k_frequent"
        desc = (f"takes a list of integers and returns the {kk} most frequent values as a list sorted by descending "
                f"frequency, ties broken by {'larger' if larger else 'smaller'} value first")
        cases = [([rng.randint(0, 6) for _ in range(rng.randint(kk, 12))],) for _ in range(6)]
        from collections import Counter
        exp = [[v for v, _ in sorted(Counter(c[0]).items(), key=lambda t: (-t[1], -t[0] if larger else t[0]))[:kk]] for c in cases]
    elif k == 2:
        base = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 12, 16])
        name, desc = "to_base", f"converts a non-negative integer to its base-{base} representation as a string using digits 0-9 then lowercase letters"
        nums = [rng.randint(0, 5000) for _ in range(6)]
        def conv(n: int) -> str:
            out = ""
            while True:
                out, n = "0123456789abcdef"[n % base] + out, n // base
                if n == 0:
                    return out
        cases, exp = [(n,) for n in nums], [conv(n) for n in nums]
    elif k == 3:
        width = rng.randint(4, 12)
        name, desc = "wrap_words", f"takes a string of words separated by single spaces and greedily wraps it into lines of at most {width} characters (a longer single word gets its own line), returning the list of lines"
        cases = [(" ".join("".join(rng.choice("abcde") for _ in range(rng.randint(1, 6))) for _ in range(rng.randint(1, 6))),) for _ in range(6)]
        def wrap(s: str) -> list[str]:
            lines: list[str] = []
            cur = ""
            for w in s.split(" "):
                if not cur:
                    cur = w
                elif len(cur) + 1 + len(w) <= width:
                    cur += " " + w
                else:
                    lines.append(cur)
                    cur = w
            return lines + ([cur] if cur else [])
        exp = [wrap(c[0]) for c in cases]
    elif k == 4:
        w, fn = rng.randint(2, 5), rng.choice(["max", "min"])
        name = f"moving_{fn}"
        buggy = f"def {name}(xs):\n    # {fn}imum of every window of {w} consecutive elements\n    return [{fn}(xs[i:i + {w}]) for i in range(len(xs) - {w})]\n"
        desc = f"is the corrected version of this buggy function (it misses the last window):\n```python\n{buggy}```\nIt must return [] when the list is shorter than {w}"
        cases = [([rng.randint(0, 20) for _ in range(rng.randint(0, 9))],) for _ in range(6)]
        f = max if fn == "max" else min
        exp = [[f(c[0][j:j + w]) for j in range(len(c[0]) - w + 1)] for c in cases]
    elif k == 5:
        shift = rng.randint(1, 25)
        name, desc = "caesar", f"shifts every ASCII letter forward by {shift} positions in the alphabet (wrapping around, preserving case) and leaves other characters unchanged"
        cases = [("".join(rng.choice("abcxyzABCXYZ !,") for _ in range(rng.randint(0, 12))),) for _ in range(6)]
        def caesar(s: str) -> str:
            return "".join(chr((ord(ch) - b + shift) % 26 + b) if ch.isascii() and ch.isalpha() and (b := 97 if ch.islower() else 65) else ch for ch in s)
        exp = [caesar(c[0]) for c in cases]
    elif k == 6:
        size = rng.randint(2, 6)
        name, desc = "chunk", f"splits a list into consecutive chunks of size {size} (the last chunk may be shorter) and returns the list of chunks"
        cases = [([rng.randint(0, 9) for _ in range(rng.randint(0, 13))],) for _ in range(6)]
        exp = [[c[0][j:j + size] for j in range(0, len(c[0]), size)] for c in cases]
    elif k == 7:
        power = rng.randint(1, 5)
        name, desc = "digit_power_sum", f"takes a non-negative integer and returns the sum of each of its decimal digits raised to the power {power}"
        nums = [rng.randint(0, 99999) for _ in range(6)]
        cases, exp = [(n,) for n in nums], [sum(int(d) ** power for d in str(n)) for n in nums]
    elif k == 8:
        length = rng.randint(2, 5)
        name, desc = "count_pal_substrings", f"returns how many substrings of length exactly {length} (counting every position) of the input string are palindromes"
        cases = [("".join(rng.choice("aab") for _ in range(rng.randint(0, 12))),) for _ in range(6)]
        exp = [sum(c[0][j:j + length] == c[0][j:j + length][::-1] for j in range(len(c[0]) - length + 1)) for c in cases]
    elif k == 9:
        sets = rng.choice(["()", "()[]", "()[]{}", "()<>", "[]{}<>"])
        name, desc = "balanced", f"takes a string made only of the characters {sets} and returns True if every bracket is balanced and properly nested"
        cases = [("".join(rng.choice(sets) for _ in range(rng.randint(0, 8))),) for _ in range(5)] + [(sets[0::2] + sets[1::2][::-1],)]
        pairs = {sets[j + 1]: sets[j] for j in range(0, len(sets), 2)}
        def ok(s: str) -> bool:
            st: list[str] = []
            for ch in s:
                if ch in pairs.values():
                    st.append(ch)
                elif not st or st.pop() != pairs[ch]:
                    return False
            return not st
        exp = [ok(c[0]) for c in cases]
    elif k == 10:
        side = rng.choice(["left", "right"])
        name, desc = "insert_position", f"takes a sorted list of integers (may contain duplicates) and a target and returns the {side}most index at which the target could be inserted to keep the list sorted"
        import bisect
        cases = [(sorted(rng.choice(range(-6, 6)) for _ in range(rng.randint(0, 8))), rng.randint(-7, 7)) for _ in range(6)]
        f2 = bisect.bisect_left if side == "left" else bisect.bisect_right
        exp = [f2(c[0], c[1]) for c in cases]
    else:
        clockwise = rng.random() < 0.5
        name = "spiral_order"
        desc = f"takes a rectangular matrix (list of lists) and returns its elements in {'clockwise' if clockwise else 'counter-clockwise'} spiral order starting from the top-left element"
        cases = []
        for _ in range(5):
            r, c = rng.randint(1, 4), rng.randint(1, 4)
            cases.append(([[rng.randint(0, 9) for _ in range(c)] for _ in range(r)],))
        def spiral(m: list[list[int]]) -> list[int]:
            m = [row[:] for row in m] if clockwise else [list(r) for r in zip(*m)]
            out: list[int] = []
            while m:
                out += m.pop(0)
                m = [list(r) for r in zip(*m)][::-1]
            return out
        exp = [spiral(c[0]) for c in cases]
    q = f"Write a Python function named `{name}` that {desc}. Return only the code in one ```python block."
    return Task("coding", f"c{i}", [{"role": "user", "content": q}],
                lambda t, n=name, c=cases, e=exp: run_python_tests(_extract_code(t), n, c, e), 768)


# ---------------------------------------------------------------- reasoning

PEOPLE = ["Ana", "Ben", "Chen", "Dina", "Eyal", "Farah", "Guy", "Hana", "Ido", "Jun", "Kim", "Lior"]
COLORS = ["red", "blue", "green", "white"]
PETS = ["cat", "dog", "fish", "bird"]


def _seating(rng: random.Random) -> tuple[str, str]:
    names = rng.sample(PEOPLE, 4)
    truth = dict(zip(names, rng.sample(COLORS, 4)))
    perms = [dict(zip(names, p)) for p in itertools.permutations(COLORS)]
    clues: list[tuple[str, Callable[[dict[str, str]], bool]]] = []
    pool: list[tuple[str, Callable[[dict[str, str]], bool]]] = []
    for n in names:
        for c in COLORS:
            if truth[n] == c:
                pool.append((f"{n}'s house is {c}.", lambda a, n=n, c=c: a[n] == c))
            else:
                pool.append((f"{n}'s house is not {c}.", lambda a, n=n, c=c: a[n] != c))
    rng.shuffle(pool)
    pool.sort(key=lambda cl: " is not " not in cl[0])  # prefer negative clues first
    target = rng.choice(names)
    for clue in pool:
        alive = [p for p in perms if all(f(p) for _, f in clues)]
        if len({p[target] for p in alive}) == 1:
            break
        if any(not clue[1](p) for p in alive):
            clues.append(clue)
    alive = [p for p in perms if all(f(p) for _, f in clues)]
    if {p[target] for p in alive} != {truth[target]}:
        raise AssertionError("seating puzzle not uniquely solvable")
    text = " ".join(c for c, _ in clues)
    return f"Four people each own one house of a different color (red, blue, green, white). {text} What color is {target}'s house? Reply with the color only.", truth[target]


def _reasoning(rng: random.Random, i: int, split: str) -> Task:
    k = i % 4
    if k == 0:
        q, ans = _seating(rng)
        return Task("reasoning", f"r{i}", [{"role": "user", "content": q}],
                    lambda t, a=ans: [c for c in COLORS if c in t.lower().split("\n")[-1]] == [a], 1024, reference=ans)
    if k == 1:
        names = rng.sample(PEOPLE, 6)
        order = names[:]
        facts = [f"{order[j]} finished before {order[j + 1]}." for j in range(5)]
        rng.shuffle(facts)
        pos = rng.randint(2, 5)
        ans = order[pos - 1]
        q = " ".join(facts) + f" There were no ties. Who finished in position {pos}? Reply with the name only."
        return Task("reasoning", f"r{i}", [{"role": "user", "content": q}],
                    lambda t, a=ans, ns=names: (last := t.strip().split("\n")[-1]) != "" and a in last and sum(n in last for n in ns) == 1, 1024, reference=ans)
    if k == 2:
        a, b = rng.randint(3, 9), rng.randint(3, 9)
        total_heads, legs = a + b, 2 * a + 4 * b
        q = f"A farm has chickens and rabbits. There are {total_heads} heads and {legs} legs in total. How many rabbits are there?"
        return Task("reasoning", f"r{i}", [{"role": "user", "content": q + ANSWER_SUFFIX}], _num_check(b), 768, reference=str(b))
    start, steps = rng.randint(1, 12), [rng.choice(["+", "*", "-"]) + str(rng.randint(2, 9)) for _ in range(5)]
    val = start
    for s in steps:
        val = val + int(s[1:]) if s[0] == "+" else val * int(s[1:]) if s[0] == "*" else val - int(s[1:])
    words = {"+": "add", "*": "multiply by", "-": "subtract"}
    q = f"Start with {start}. Then " + ", then ".join(f"{words[s[0]]} {s[1:]}" for s in steps) + ". What is the result?"
    return Task("reasoning", f"r{i}", [{"role": "user", "content": q + ANSWER_SUFFIX}], _num_check(val), 768, reference=str(val))


# ---------------------------------------------------------------- Hebrew

HE_WORDS = [
    ("dog", ["כלב"]), ("house", ["בית"]), ("water", ["מים"]), ("book", ["ספר"]), ("sun", ["שמש"]), ("tree", ["עץ"]),
    ("bread", ["לחם"]), ("city", ["עיר"]), ("sea", ["ים"]), ("moon", ["ירח"]), ("school", ["בית ספר", "בית-ספר"]),
    ("teacher", ["מורה"]), ("window", ["חלון"]), ("door", ["דלת"]), ("chair", ["כיסא", "כסא"]), ("table", ["שולחן"]),
    ("milk", ["חלב"]), ("apple", ["תפוח"]), ("road", ["כביש", "דרך"]), ("car", ["מכונית", "רכב"]), ("friend", ["חבר"]),
    ("family", ["משפחה"]), ("work", ["עבודה"]), ("time", ["זמן"]), ("year", ["שנה"]), ("day", ["יום"]), ("night", ["לילה"]),
    ("river", ["נהר"]), ("mountain", ["הר"]), ("flower", ["פרח"]), ("bird", ["ציפור"]), ("fish", ["דג"]), ("heart", ["לב"]),
    ("hand", ["יד"]), ("eye", ["עין"]), ("head", ["ראש"]), ("music", ["מוזיקה"]), ("language", ["שפה"]), ("money", ["כסף"]),
    ("question", ["שאלה"]), ("answer", ["תשובה"]), ("world", ["עולם"]), ("child", ["ילד"]), ("woman", ["אישה", "אשה"]),
    ("man", ["איש", "גבר"]), ("king", ["מלך"]), ("war", ["מלחמה"]), ("peace", ["שלום"]),
]
HE_PEOPLE = ["דנה", "יוסי", "מיכל", "אורי", "נועה", "עמית", "שירה", "רון", "טל", "גיא", "הדר", "ליאור"]


def _hebrew(rng: random.Random, i: int, split: str) -> Task:
    k = i % 4
    if k == 0:
        en, he = rng.choice(_pool(HE_WORDS, split))
        q = f"תרגם לעברית את המילה האנגלית '{en}'. ענה במילה אחת בעברית בלבד."
        return Task("hebrew", f"h{i}", [{"role": "user", "content": q}], lambda t, a=he: any(x in t for x in a), 64, reference=he[0])
    if k == 1:
        a, b = rng.sample(HE_PEOPLE, 2)
        x, y, p = rng.randint(20, 90), rng.randint(3, 15), rng.choice([2, 3, 4])
        ans = (x - y) * p
        q = f"ל{a} היו {x} שקלים. {a} קנה מתנה ב-{y} שקלים, ואז {b} נתן ל{a} פי {p} מהסכום שנשאר. כמה שקלים קיבל {a} מ{b}?" + HE_ANSWER_SUFFIX
        return Task("hebrew", f"h{i}", [{"role": "user", "content": q}], _num_check(ans), 768, reference=str(ans))
    if k == 2:
        country, answers = rng.choice(_pool(HE_CAPITALS, split))
        q = f"מהי עיר הבירה של {country}? ענה בשם העיר בלבד, בעברית."
        return Task("hebrew", f"h{i}", [{"role": "user", "content": q}], lambda t, a=answers: any(x in t for x in a), 64, reference=answers[0])
    names = rng.sample(HE_PEOPLE, 3)
    cities = rng.sample(["חיפה", "אילת", "ירושלים", "טבריה", "באר שבע", "נתניה", "עכו"], 3)
    hobbies = rng.sample(["שחייה", "ציור", "שחמט", "ריצה", "בישול"], 3)
    passage = " ".join(f"{n} גר/ה ב{c} ואוהב/ת {h}." for n, c, h in zip(names, cities, hobbies))
    j = rng.randrange(3)
    q = f"{passage}\nמי מהם גר/ה ב{cities[j]}? ענה בשם בלבד."
    return Task("hebrew", f"h{i}", [{"role": "user", "content": q}],
                lambda t, a=names[j], ns=names: a in t and sum(n in t for n in ns) == 1, 64, reference=names[j])


# ---------------------------------------------------------------- English

def _english(rng: random.Random, i: int, split: str) -> Task:
    names = rng.sample(PEOPLE, 4)
    times = sorted(rng.sample(range(7, 20), 4))
    order = rng.sample(range(4), 4)
    arrivals = {names[order[j]]: times[j] for j in range(4)}
    sentences = [f"{n} arrived at {h}:00." for n, h in arrivals.items()]
    rng.shuffle(sentences)
    k = i % 3
    if k == 0:
        ans = sorted(arrivals, key=arrivals.get)[1]
        q = " ".join(sentences) + " Who arrived second? Reply with the name only."
        check = lambda t, a=ans, ns=names: a in t and sum(n in t for n in ns) == 1
    elif k == 1:
        n1, n2 = rng.sample(names, 2)
        ans = str(abs(arrivals[n1] - arrivals[n2]))
        q = " ".join(sentences) + f" How many hours apart did {n1} and {n2} arrive?" + ANSWER_SUFFIX
        check = _num_check(float(ans))
    else:
        ans = "yes" if arrivals[names[0]] < arrivals[names[1]] else "no"
        q = " ".join(sentences) + f" Did {names[0]} arrive before {names[1]}? Answer yes or no."
        check = lambda t, a=ans: t.strip().lower().strip(".!* ").startswith(a)
    return Task("english", f"e{i}", [{"role": "user", "content": q}], check, 512, reference=ans)


# ---------------------------------------------------------------- instruction following

TOPICS = ["sleep", "rivers", "bicycles", "tea", "volcanoes", "libraries", "deserts", "pianos", "honey", "bridges",
          "owls", "glaciers", "chess", "lighthouses", "coffee", "maps", "kites", "coral reefs", "trains", "clocks", "bees"]


def _instruction(rng: random.Random, i: int, split: str) -> Task:
    topic = rng.choice(_pool(TOPICS, split))
    n = rng.randint(3, 6)
    word = rng.choice(["indeed", "notably", "quietly", "remarkably"])
    k = i % 5
    if k == 0:
        q = f"Write exactly {n} bullet points about {topic}, each starting with '- ', all in lowercase, with no other text."
        check = lambda t, n=n: (ls := [x for x in t.strip().splitlines() if x.strip()]) != [] and len(ls) == n and all(x.lstrip().startswith("- ") for x in ls) and t == t.lower()
    elif k == 1:
        q = f"Write a paragraph about {topic} of between {n * 10} and {n * 10 + 20} words that ends with the word '{word}'."
        check = lambda t, n=n, w=word: n * 10 <= len(t.split()) <= n * 10 + 20 and t.strip().rstrip(".!\"'").lower().endswith(w)
    elif k == 2:
        q = (f"Return a JSON array of {n} objects about {topic}; each object has keys \"id\" (integers 1..{n} in order) "
             f"and \"fact\" (a string). Output JSON only.")
        def check(t: str, n: int = n) -> bool:
            body = t.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                arr = json.loads(body)
            except ValueError:
                return False
            return isinstance(arr, list) and len(arr) == n and all(isinstance(o, dict) and set(o) == {"id", "fact"} and o["id"] == j + 1 and isinstance(o["fact"], str) for j, o in enumerate(arr))
    elif k == 3:
        q = f"Write {n} sentences about {topic}. Do not use the letter 'e' anywhere in your answer."
        check = lambda t: len(t.strip()) > 20 and "e" not in t.lower()
    else:
        q = f"Answer in exactly two lines: first line the word '{topic.upper()}' in capitals, second line one sentence about it in Hebrew."
        check = lambda t, tp=topic.upper(): (ls := [x for x in t.strip().splitlines() if x.strip()]) != [] and len(ls) == 2 and ls[0].strip().strip("*") == tp and bool(re.search("[֐-׿]", ls[1]))
    return Task("instruction_following", f"i{i}", [{"role": "user", "content": q}], check, 512)


# ---------------------------------------------------------------- tools and agentic

TOOLS = {
    "get_weather": {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
    "convert_currency": {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount between ISO currency codes", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_code": {"type": "string"}, "to_code": {"type": "string"}}, "required": ["amount", "from_code", "to_code"]}}},
    "search_flights": {"type": "function", "function": {"name": "search_flights", "description": "Find flights between two airports on a date (YYYY-MM-DD)", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}},
    "get_price": {"type": "function", "function": {"name": "get_price", "description": "Unit price in USD of a catalog item", "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}},
    "get_stock": {"type": "function", "function": {"name": "get_stock", "description": "Units in stock for a catalog item", "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}}},
}
CITIES = ["Haifa", "Lisbon", "Osaka", "Nairobi", "Lima", "Oslo", "Quito", "Hanoi", "Perth", "Tunis", "Riga", "Cusco"]
ITEMS = ["lamp", "kettle", "desk", "blender", "mirror", "router", "stool", "fan", "heater", "toaster", "monitor", "printer", "backpack", "drill", "speaker", "tent"]


def _tool(rng: random.Random, i: int, split: str) -> Task:
    tools = list(TOOLS.values())
    k = i % 3
    if k == 0:
        city, unit = rng.choice(_pool(CITIES, split)), rng.choice(["c", "f"])
        q = f"How warm is it in {city} at the moment? Give it in {'fahrenheit' if unit == 'f' else 'celsius'}."
        check = lambda t, c=city, u=unit: any(x["name"] == "get_weather" and str(x["arguments"].get("city", "")).lower() == c.lower() and x["arguments"].get("unit", "c") == u for x in parse_tool_calls(t))
    elif k == 1:
        amt, a, b = rng.randint(10, 999), *rng.sample(["USD", "EUR", "ILS", "JPY", "GBP"], 2)
        q = f"How much is {amt} {a} in {b}?"
        check = lambda t, m=amt, a=a, b=b: any(x["name"] == "convert_currency" and x["arguments"].get("amount") == m and x["arguments"].get("from_code") == a and x["arguments"].get("to_code") == b for x in parse_tool_calls(t))
    else:
        o, d = rng.sample(["TLV", "LHR", "JFK", "CDG", "NRT", "ATH"], 2)
        day = rng.randint(1, 28)
        q = f"Find me flights from {o} to {d} on March {day}, 2027."
        check = lambda t, o=o, d=d, ds=f"2027-03-{day:02d}": any(x["name"] == "search_flights" and x["arguments"].get("origin") == o and x["arguments"].get("destination") == d and x["arguments"].get("date") == ds for x in parse_tool_calls(t))
    return Task("tool_use", f"t{i}", [{"role": "user", "content": q}], check, 256, tools=tools)


def _agentic(rng: random.Random, i: int, split: str) -> Task:
    items = rng.sample(ITEMS, 2)
    prices = {it: rng.randint(5, 80) for it in items}
    stock = {it: rng.randint(0, 20) for it in items}
    want = rng.randint(2, 12)
    q = (f"I want to buy {want} units each of the {items[0]} and the {items[1]}, but only as many as are in stock. "
         f"Use the tools to check prices and stock, then tell me the total cost in USD." + ANSWER_SUFFIX)
    total = sum(prices[it] * min(want, stock[it]) for it in items)

    def env(name: str, args: dict[str, Any]) -> str:
        item = str(args.get("item", "")).lower().strip()
        if name == "get_price" and item in prices:
            return json.dumps({"item": item, "unit_price_usd": prices[item]})
        if name == "get_stock" and item in stock:
            return json.dumps({"item": item, "in_stock": stock[item]})
        return json.dumps({"error": "unknown item or tool"})

    return Task("agentic", f"a{i}", [{"role": "user", "content": q}], _num_check(total), 512,
                tools=[TOOLS["get_price"], TOOLS["get_stock"], TOOLS["get_weather"]], reference=str(total), meta={"env": env})


# ---------------------------------------------------------------- long context and multi-turn

FILLER_SENTENCES = [
    "The committee reviewed the quarterly logistics report and noted minor delays at the northern depot.",
    "Weather conditions remained stable throughout the week, with light winds from the west.",
    "Several volunteers repainted the community hall and replaced two broken windows.",
    "The library extended its weekend hours to accommodate students preparing for exams.",
    "A local bakery introduced a new rye loaf that quickly became popular with commuters.",
    "Engineers inspected the old bridge and recommended routine maintenance in the spring.",
    "The orchestra rehearsed a new arrangement of a familiar folk melody.",
    "Traffic on the coastal road was lighter than usual due to the holiday.",
]


def _long(rng: random.Random, i: int, split: str) -> Task:
    n_sent = rng.randint(80, 220)
    body = [rng.choice(FILLER_SENTENCES) for _ in range(n_sent)]
    key = f"{rng.choice(['amber', 'cobalt', 'maple', 'quartz', 'cedar'])}-{rng.randint(1000, 9999)}"
    other = f"{rng.choice(['onyx', 'birch', 'slate'])}-{rng.randint(1000, 9999)}"
    owner = rng.choice(PEOPLE)
    decoy = rng.choice([p for p in PEOPLE if p != owner])
    body.insert(rng.randrange(n_sent), f"Note: the access code assigned to {owner} is {key}.")
    body.insert(rng.randrange(n_sent), f"Note: the access code assigned to {decoy} is {other}.")
    q = " ".join(body) + f"\n\nWhat is the access code assigned to {owner}? Reply with the code only."
    return Task("long_context", f"l{i}", [{"role": "user", "content": q}], lambda t, a=key, b=other: a in t and b not in t, 32, reference=key)


def _multi(rng: random.Random, i: int, split: str) -> Task:
    k = i % 3
    if k == 0:
        old, new = rng.randint(10, 99), rng.randint(10, 99)
        msgs = [{"role": "user", "content": f"My table number is {old}."}, {"role": "assistant", "content": "Got it."},
                {"role": "user", "content": "Can you recommend a dessert?"}, {"role": "assistant", "content": "A lemon tart is a good choice."},
                {"role": "user", "content": f"Correction: my table number is actually {new}, not {old}."}, {"role": "assistant", "content": "Thanks, updated."},
                {"role": "user", "content": "Which table number should the waiter go to? Number only."}]
        return Task("multi_turn", f"t{i}", msgs, lambda t, n=new, o=old: str(n) in t and str(o) not in t, 32, reference=str(new))
    if k == 1:
        topic = rng.choice(TOPICS)
        msgs = [{"role": "user", "content": "From now on, answer every message in ALL CAPITAL LETTERS."}, {"role": "assistant", "content": "UNDERSTOOD."},
                {"role": "user", "content": "What is 3 plus 4?"}, {"role": "assistant", "content": "7."},
                {"role": "user", "content": f"Tell me one fact about {topic}."}]
        return Task("multi_turn", f"t{i}", msgs, lambda t: len(re.findall("[a-z]", t)) == 0 and len(re.findall("[A-Z]", t)) > 5, 128)
    a, b, c = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 5)
    msgs = [{"role": "user", "content": f"Let x = {a}."}, {"role": "assistant", "content": f"OK, x = {a}."},
            {"role": "user", "content": f"Let y = x + {b}."}, {"role": "assistant", "content": f"OK, y = {a + b}."},
            {"role": "user", "content": f"Now multiply y by {c} and subtract x. What do you get?" + ANSWER_SUFFIX}]
    return Task("multi_turn", f"t{i}", msgs, _num_check((a + b) * c - a), 256, reference=str((a + b) * c - a))


# ---------------------------------------------------------------- factual and general

FACTS = [
    ("What is the chemical symbol for gold?", ["au"]), ("How many planets are in the Solar System?", ["8", "eight"]),
    ("Who wrote 'Pride and Prejudice'?", ["austen"]), ("What is the largest ocean on Earth?", ["pacific"]),
    ("What is the boiling point of water at sea level in Celsius?", ["100"]), ("Which planet is known as the Red Planet?", ["mars"]),
    ("What gas do plants absorb for photosynthesis?", ["carbon dioxide", "co2"]), ("Who developed the theory of general relativity?", ["einstein"]),
    ("What is the capital of Canada?", ["ottawa"]), ("What is the hardest natural substance?", ["diamond"]),
    ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]), ("What is the chemical symbol for sodium?", ["na"]),
    ("In what year did World War II end?", ["1945"]), ("What is the largest planet in the Solar System?", ["jupiter"]),
    ("Who wrote 'Hamlet'?", ["shakespeare"]), ("What is the longest river in South America?", ["amazon"]),
    ("What is the smallest prime number?", ["2", "two"]), ("What is the currency of Japan?", ["yen"]),
    ("Who was the first person to walk on the Moon?", ["armstrong"]), ("What is the capital of Australia?", ["canberra"]),
    ("How many bones are in the adult human body?", ["206"]), ("What organ pumps blood through the body?", ["heart"]),
    ("Which element has atomic number 1?", ["hydrogen"]), ("Who proposed the theory of evolution by natural selection?", ["darwin"]),
    ("What is the tallest mountain on Earth above sea level?", ["everest"]), ("What language is primarily spoken in Brazil?", ["portuguese"]),
    ("What is the freezing point of water in Fahrenheit?", ["32"]), ("Who composed the Ninth Symphony with the 'Ode to Joy'?", ["beethoven"]),
    ("What is the largest desert in the world by area, including polar deserts?", ["antarctic"]), ("Which planet has the most prominent rings?", ["saturn"]),
    ("What is the capital of Turkey?", ["ankara"]), ("What is the speed of light in vacuum, approximately, in km per second?", ["300,000", "300000", "299,792", "299792"]),
    ("Who invented the World Wide Web?", ["berners-lee", "berners lee"]), ("What is the main language of Iran?", ["persian", "farsi"]),
    ("What is the chemical formula of table salt?", ["nacl"]), ("Which country has the largest population as of 2024?", ["india"]),
    ("What is the square of 17?", ["289"]), ("Who wrote 'One Hundred Years of Solitude'?", ["garcía márquez", "garcia marquez", "márquez", "marquez"]),
    ("What is the capital of Kenya?", ["nairobi"]), ("Which metal is liquid at room temperature?", ["mercury"]),
    ("What is the largest bone in the human body?", ["femur"]), ("Who painted 'The Starry Night'?", ["van gogh"]),
    ("What is the capital of Egypt?", ["cairo"]), ("How many continents are there?", ["7", "seven"]),
    ("What is the powerhouse of the cell?", ["mitochondri"]), ("Which ocean lies between Africa and Australia?", ["indian"]),
    ("What is the capital of Argentina?", ["buenos aires"]), ("Who discovered penicillin?", ["fleming"]),
    ("What year did the Berlin Wall fall?", ["1989"]), ("What is the chemical symbol for iron?", ["fe"]),
    ("Which instrument has 88 keys?", ["piano"]), ("What is the capital of South Korea?", ["seoul"]),
    ("What is the largest mammal?", ["blue whale"]), ("Who wrote 'The Odyssey'?", ["homer"]),
    ("What is the capital of Norway?", ["oslo"]), ("What is the main ingredient in guacamole?", ["avocado"]),
    ("What is 2 to the power of 10?", ["1024"]), ("Which gas makes up most of Earth's atmosphere?", ["nitrogen"]),
    ("What is the capital of Peru?", ["lima"]), ("Who was the first president of the United States?", ["washington"]),
    ("What is the national language of Israel besides Arabic, which has special status?", ["hebrew"]), ("What is the capital of Vietnam?", ["hanoi"]),
    ("How many sides does a pentagon have?", ["5", "five"]), ("Which scientist formulated the laws of motion and universal gravitation?", ["newton"]),
    ("What is the capital of Portugal?", ["lisbon"]), ("Which blood type is the universal donor?", ["o negative", "o-", "o−"]),
    ("What is the longest river in Africa?", ["nile"]), ("Who wrote '1984'?", ["orwell"]),
    ("What is the capital of Greece?", ["athens"]), ("What is the SI unit of electric current?", ["ampere", "amp"]),
    ("Which country gifted the Statue of Liberty to the USA?", ["france"]), ("What is the capital of Mexico?", ["mexico city"]),
    ("Which planet is closest to the Sun?", ["mercury"]), ("What is the largest country by area?", ["russia"]),
    ("Who developed the polio vaccine first licensed in 1955?", ["salk"]), ("What is the capital of Thailand?", ["bangkok"]),
    ("What is the chemical symbol for potassium?", ["k"]), ("What is the capital of Spain?", ["madrid"]),
    ("How many minutes are in a day?", ["1440", "1,440"]), ("What is the capital of Ireland?", ["dublin"]),
    ("Who wrote 'War and Peace'?", ["tolstoy"]), ("What is the capital of Chile?", ["santiago"]),
    ("Which vitamin does the body produce in sunlight?", ["vitamin d", " d"]), ("What is the capital of Sweden?", ["stockholm"]),
    ("Which animal is known as the king of the jungle?", ["lion"]), ("What is the capital of Morocco?", ["rabat"]),
    ("What is the largest internal organ of the human body?", ["liver"]), ("What is the capital of Poland?", ["warsaw"]),
]


def _factual(rng: random.Random, i: int, split: str) -> Task:
    pool = _pool(FACTS, split)
    q, ans = pool[i % len(pool)]
    check = (lambda t, a=ans: bool(re.search(r"\b" + re.escape(a[0]) + r"\b", t.lower()))) if len(ans[0]) <= 2 else (lambda t, a=ans: any(x in t.lower() for x in a))
    return Task("factual", f"f{i}", [{"role": "user", "content": q + " Answer briefly."}], check, 96, reference=ans[0])


def _general(rng: random.Random, i: int, split: str) -> Task:
    k = i % 3
    if k == 0:
        item, price, qty = rng.choice(_pool(ITEMS, split)), rng.randint(20, 90), rng.randint(2, 9)
        disc = rng.choice([5, 10, 15])
        total = price * qty * (100 - disc) / 100
        q = f"Write a short polite email ordering {qty} units of the {item} at ${price} each with a {disc}% discount. State the final total."
        tot = f"{total:.2f}".rstrip("0").rstrip(".")
        return Task("general_assistance", f"g{i}", [{"role": "user", "content": q}],
                    lambda t, a=tot, b=f"{total:,.2f}", it=item: (a in t.replace(",", "") or b in t) and it in t.lower(), 512)
    if k == 1:
        city, n = rng.choice(_pool(CITIES, split)), rng.randint(2, 5)
        q = f"Make a {n}-day itinerary for {city}. Use exactly the headings 'Day 1' to 'Day {n}', each followed by at most 3 bullet points."
        return Task("general_assistance", f"g{i}", [{"role": "user", "content": q}],
                    lambda t, n=n: all(f"day {d}" in t.lower() for d in range(1, n + 1)) and f"day {n + 1}" not in t.lower(), 1024)
    name, product, issue = rng.choice(PEOPLE), rng.choice(ITEMS), rng.choice(["arrived broken", "is missing a part", "stopped working"])
    q = (f"Summarize this complaint in one sentence of at most 25 words that mentions the customer's name and the product: "
         f"'Hello, this is {name}. I bought a {product} last week and it {issue}. I have tried contacting support twice "
         f"without a reply and I would like a replacement or a refund as soon as possible.'")
    return Task("general_assistance", f"g{i}", [{"role": "user", "content": q}],
                lambda t, n=name, p=product: n in t and p in t.lower() and len(t.split()) <= 25 and t.count(".") <= 2, 128)


GENERATORS: dict[str, Callable[[random.Random, int, str], Task]] = {
    "math": _math, "coding": _coding, "reasoning": _reasoning, "hebrew": _hebrew, "english": _english,
    "instruction_following": _instruction, "tool_use": _tool, "long_context": _long, "multi_turn": _multi,
    "factual": _factual, "general_assistance": _general, "agentic": _agentic,
}


POOL_PARTITIONED = {"factual"}  # finite pools split into exact thirds by _pool


def split_of(messages: list[dict[str, Any]]) -> str:
    """Hash partition: every distinct prompt belongs to exactly one split."""
    import hashlib

    digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).digest()
    return ("train", "validation", "test")[digest[0] % 3]


def build_tasks(split: str, per_category: int = 30, categories: tuple[str, ...] = CATEGORIES) -> list[Task]:
    """Unique prompts per split; splits are disjoint by hash (or by exact pool thirds)."""
    if split not in SPLIT_SEEDS:
        raise ValueError("split must be train, validation or test")
    tasks: list[Task] = []
    for c_index, category in enumerate(categories):
        rng = random.Random(SPLIT_SEEDS[split] + 1000 * c_index)
        seen: set[str] = set()
        made: list[Task] = []
        attempt = 0
        while len(made) < per_category and attempt < per_category * 400:
            task = GENERATORS[category](rng, attempt, split)
            attempt += 1
            key = json.dumps(task.messages, ensure_ascii=False, sort_keys=True)
            if key in seen or (category not in POOL_PARTITIONED and split_of(task.messages) != split):
                continue
            seen.add(key)
            task.task_id = f"{category[:3]}{len(made)}"
            made.append(task)
        tasks += made
    return tasks


ChatFn = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None, int], str]


def run_task(chat: ChatFn, task: Task, max_tool_rounds: int = 4) -> str:
    """Single reply, or a bounded tool loop when the task supplies a simulated environment."""
    env = task.meta.get("env")
    messages = list(task.messages)
    reply = chat(messages, task.tools, task.max_tokens)
    if env is None:
        return reply
    for _ in range(max_tool_rounds):
        calls = parse_tool_calls(reply)
        if not calls:
            return reply
        messages.append({"role": "assistant", "content": reply})
        for call in calls[:4]:
            messages.append({"role": "tool", "content": env(call["name"], call["arguments"])})
        reply = chat(messages, task.tools, task.max_tokens)
    return reply


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(chat: ChatFn, split: str = "validation", per_category: int = 30,
             categories: tuple[str, ...] = CATEGORIES, log: Callable[[str], None] | None = None,
             start_index: int = 0) -> dict[str, Any]:
    """Score items [start_index, per_category) of every category (prefixes are stable)."""
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for task in build_tasks(split, per_category, categories):
        if int(re.sub(r"\D", "", task.task_id) or 0) < start_index:
            continue
        t0 = time.perf_counter()
        try:
            reply = run_task(chat, task)
            ok = bool(task.check(reply))
        except Exception as exc:  # a crashing item is a failed item, recorded
            reply, ok = f"[error] {type(exc).__name__}: {exc}", False
        rows.append({"category": task.category, "id": task.task_id, "passed": ok, "seconds": time.perf_counter() - t0,
                     "reply": reply, "reference": task.reference})
        if log:
            log(f"{task.task_id} {'PASS' if ok else 'fail'}")
    per: dict[str, Any] = {}
    for c in categories:
        k = sum(r["passed"] for r in rows if r["category"] == c)
        n = sum(r["category"] == c for r in rows)
        per[c] = {"score": k / n if n else 0.0, "passed": k, "n": n, "ci95": wilson(k, n)}
    total_k, total_n = sum(r["passed"] for r in rows), len(rows)
    return {"suite_version": SUITE_VERSION, "split": split, "per_category": per_category, "categories": per,
            "scores": {c: v["score"] for c, v in per.items()}, "mean": sum(v["score"] for v in per.values()) / len(per),
            "pooled": {"passed": total_k, "n": total_n, "ci95": wilson(total_k, total_n)},
            "meets_target": {c: per[c]["score"] >= TARGET_THRESHOLDS[c] for c in categories},
            "target_profile": "provisional ModelForge thresholds on suite v2; not a measured GPT-5.6 Luna comparison",
            "seconds": time.perf_counter() - start, "rows": rows}


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Paired item-level comparison with an exact two-sided sign test."""
    base = {(r["category"], r["id"]): r["passed"] for r in baseline["rows"]}
    cand = {(r["category"], r["id"]): r["passed"] for r in candidate["rows"]}
    if set(base) != set(cand):
        raise ValueError("Runs must cover identical items")

    def sign_test(wins: int, losses: int) -> float:
        n = wins + losses
        if n == 0:
            return 1.0
        tail = sum(math.comb(n, j) for j in range(0, min(wins, losses) + 1)) / 2**n
        return min(1.0, 2 * tail)

    out: dict[str, Any] = {}
    for c in sorted({k[0] for k in base} | {"__all__"}):
        keys = [k for k in base if c == "__all__" or k[0] == c]
        wins = sum(cand[k] and not base[k] for k in keys)
        losses = sum(base[k] and not cand[k] for k in keys)
        out[c] = {"wins": wins, "losses": losses, "n": len(keys), "p_value": sign_test(wins, losses),
                  "delta": (wins - losses) / len(keys) if keys else 0.0}
    out["__all__"]["significant_improvement"] = out["__all__"]["delta"] > 0 and out["__all__"]["p_value"] < 0.05
    out["__all__"]["regressed_categories"] = [c for c, v in out.items() if c != "__all__" and v["losses"] - v["wins"] >= 3]
    return out
