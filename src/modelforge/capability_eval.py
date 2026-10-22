# ruff: noqa: E731
# mypy: disable-error-code="misc"
"""Contamination-resistant capability evaluation for local chat models.

Tasks are generated procedurally from a seed (or are short project-written
items), scored by deterministic checkers, and split into disjoint validation and
test seeds. Validation drives model selection; the test split is reserved for
final reporting. Nothing here calls a hosted model.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

CATEGORIES = (
    "coding",
    "math",
    "reasoning",
    "hebrew",
    "english",
    "instruction_following",
    "tool_use",
    "long_conversation",
    "factual",
    "general_assistance",
)

# Provisional frontier-class profile. These are ModelForge thresholds on these
# procedural tasks, NOT a measured GPT-5.6 Luna score: no authorized access to
# that model was available, so equivalence cannot be established here.
TARGET_THRESHOLDS = {
    "coding": 0.90,
    "math": 0.90,
    "reasoning": 0.90,
    "hebrew": 0.90,
    "english": 0.95,
    "instruction_following": 0.90,
    "tool_use": 0.95,
    "long_conversation": 0.95,
    "factual": 0.90,
    "general_assistance": 0.90,
}

SPLIT_SEEDS = {"validation": 1_000, "test": 9_000_000, "train": 50_000_000}

Messages = list[dict[str, Any]]
_POOL_OFFSET = {"train": 0, "validation": 1, "test": 2}


def _pool(items: list[Any] | tuple[Any, ...], split: str) -> list[Any]:
    """Disjoint thirds of a finite item pool for train/validation/test."""
    return list(items[_POOL_OFFSET[split]::3])


@dataclass
class Task:
    category: str
    task_id: str
    messages: Messages
    check: Callable[[str], bool]
    max_tokens: int = 384
    tools: list[dict[str, Any]] | None = None
    reference: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _last_number(text: str) -> float | None:
    marked = re.findall(r"(?:answer|תשובה)\s*[:：]\s*\**\s*(-?[\d,]+(?:\.\d+)?)", text, re.I)
    found = marked or re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def _num_check(expected: float) -> Callable[[str], bool]:
    return lambda text: (v := _last_number(text)) is not None and abs(v - expected) < 1e-6


ANSWER_SUFFIX = " End your reply with a final line of the form 'Answer: <number>'."
HE_ANSWER_SUFFIX = " סיים בשורה אחרונה בפורמט 'Answer: <מספר>'."


def _math(rng: random.Random, i: int, split: str) -> Task:
    kind = i % 4
    if kind == 0:
        a, b, c, d = rng.randint(20, 90), rng.randint(3, 19), rng.randint(2, 9), rng.randint(3, 12)
        q = (f"A shop starts with {a} notebooks, sells {b}, then receives {c} boxes with {d} "
             f"notebooks each, and finally gives away a third of what it has, rounded down. How many notebooks remain?")
        total = a - b + c * d
        ans = total - total // 3
    elif kind == 1:
        x, m, k = rng.randint(-15, 15), rng.randint(2, 12), rng.randint(-40, 40)
        q = f"Solve for x: {m}x + ({k}) = {m * x + k}."
        ans = x
    elif kind == 2:
        n, p = rng.randint(10**5, 10**6), rng.choice([7, 11, 13, 17, 19, 23])
        q = f"What is the remainder when {n} is divided by {p}?"
        ans = n % p
    else:
        speed, hours, back = rng.randint(40, 90), rng.randint(2, 6), rng.randint(30, 80)
        q = (f"A car drives {hours} hours at {speed} km/h, then returns along the same road at {back} km/h. "
             f"What is the total driving time in minutes, rounded down to an integer?")
        ans = hours * 60 + (speed * hours * 60) // back
    return Task("math", f"math-{i}", [{"role": "user", "content": q + ANSWER_SUFFIX}], _num_check(ans), 512, reference=str(ans))


NAMES = ["Avi", "Bella", "Carmen", "Dov", "Eli", "Fay", "Gil", "Hila"]


def _reasoning(rng: random.Random, i: int, split: str) -> Task:
    kind = i % 3
    if kind == 0:
        people = rng.sample(NAMES, 5)
        facts = [f"{people[j]} is taller than {people[j + 1]}." for j in range(4)]
        rng.shuffle(facts)
        target = rng.choice(["tallest", "shortest"])
        ans = people[0] if target == "tallest" else people[-1]
        q = " ".join(facts) + f" Who is the {target}? Reply with the name only."
        check: Callable[[str], bool] = lambda t, a=ans: a.lower() in t.lower().split("\n")[-1].lower() and sum(n.lower() in t.lower().split("\n")[-1].lower() for n in NAMES) == 1
        return Task("reasoning", f"reasoning-{i}", [{"role": "user", "content": q}], check, 256, reference=ans)
    if kind == 1:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        start, n = rng.randrange(7), rng.randint(20, 400)
        ans = days[(start + n) % 7]
        q = f"Today is {days[start]}. What day of the week will it be in {n} days? Reply with the day name only."
        return Task("reasoning", f"reasoning-{i}", [{"role": "user", "content": q}],
                    lambda t, a=ans: max(days, key=lambda d: t.lower().rfind(d.lower())) == a and a.lower() in t.lower(), 384, reference=ans)
    word = "".join(rng.choice("abcdefghr") for _ in range(rng.randint(9, 14)))
    letter = rng.choice(sorted(set(word)))
    ans_n = word.count(letter)
    q = f"How many times does the letter '{letter}' appear in the string '{word}'?" + ANSWER_SUFFIX
    return Task("reasoning", f"reasoning-{i}", [{"role": "user", "content": q}], _num_check(ans_n), 384, reference=str(ans_n))


CODE_SPECS = [
    ("sum_even_squares", "takes a list of integers and returns the sum of the squares of the even numbers", lambda xs: sum(x * x for x in xs if x % 2 == 0), "list"),
    ("count_vowels", "takes a string and returns the number of vowels (a, e, i, o, u, case-insensitive)", lambda s: sum(c in "aeiou" for c in s.lower()), "str"),
    ("second_largest", "takes a list of integers with at least two distinct values and returns the second largest distinct value", lambda xs: sorted(set(xs))[-2], "list2"),
    ("run_length", "takes a string and returns its run-length encoding as a string like 'a3b1' for 'aaab'", lambda s: "".join(f"{c}{len(m)}" for m, c in [(m.group(0), m.group(0)[0]) for m in re.finditer(r"(.)\1*", s)]), "str"),
    ("is_palindrome", "takes a string and returns True if it reads the same backwards ignoring case and non-alphanumeric characters", lambda s: (t := [c.lower() for c in s if c.isalnum()]) == t[::-1], "pal"),
    ("rotate_list", "takes a list and an integer k and returns the list rotated to the right by k positions (k may exceed the length; empty list returns empty list)", lambda xs, k: xs[-(k % len(xs)):] + xs[:-(k % len(xs))] if xs else [], "rot"),
    ("flatten", "takes an arbitrarily nested list of integers and returns a flat list of the integers in order", None, "nested"),
    ("merge_intervals", "takes a list of [start, end] integer intervals and returns the merged, sorted list of overlapping intervals as lists", None, "intervals"),
    ("fizzbuzz_list", "takes a positive integer n and returns a list of strings for 1..n where multiples of 3 are 'Fizz', of 5 are 'Buzz', of both are 'FizzBuzz', otherwise the number as a string", lambda n: ["FizzBuzz" if k % 15 == 0 else "Fizz" if k % 3 == 0 else "Buzz" if k % 5 == 0 else str(k) for k in range(1, n + 1)], "n"),
    ("word_frequencies", "takes a string and returns a dict mapping each lowercase whitespace-separated word to its count", lambda s: {w: s.lower().split().count(w) for w in s.lower().split()}, "words"),
    ("transpose", "takes a rectangular matrix as a list of lists and returns its transpose as a list of lists", lambda m: [list(r) for r in zip(*m)], "matrix"),
    ("gcd_list", "takes a non-empty list of positive integers and returns their greatest common divisor", None, "pos"),
]


def _flatten(x: Any) -> list[int]:
    return [y for e in x for y in (_flatten(e) if isinstance(e, list) else [e])]


def _merge(iv: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _code_inputs(rng: random.Random, kind: str) -> list[tuple[Any, ...]]:
    cases: list[tuple[Any, ...]] = []
    for _ in range(6):
        if kind in {"list", "list2"}:
            xs = [rng.randint(-20, 20) for _ in range(rng.randint(2, 8))]
            if kind == "list2" and len(set(xs)) < 2:
                xs += [max(xs) + 1]
            cases.append((xs,))
        elif kind == "str":
            cases.append(("".join(rng.choice("aabbcEIou xyz") for _ in range(rng.randint(0, 12))),))
        elif kind == "pal":
            base = "".join(rng.choice("abcA, ") for _ in range(rng.randint(1, 6)))
            cases.append((base + (base[::-1] if rng.random() < 0.5 else "x"),))
        elif kind == "rot":
            cases.append(([rng.randint(0, 9) for _ in range(rng.randint(0, 6))], rng.randint(0, 10)))
        elif kind == "n":
            cases.append((rng.randint(1, 20),))
        elif kind == "words":
            cases.append((" ".join(rng.choice(["Sun", "sun", "moon", "Star", "sky"]) for _ in range(rng.randint(1, 7))),))
        elif kind == "matrix":
            r, c = rng.randint(1, 4), rng.randint(1, 4)
            cases.append(([[rng.randint(0, 9) for _ in range(c)] for _ in range(r)],))
        elif kind == "pos":
            g = rng.randint(1, 9)
            cases.append(([g * rng.randint(1, 12) for _ in range(rng.randint(1, 5))],))
        elif kind == "nested":
            cases.append(([1, [rng.randint(0, 9), [rng.randint(0, 9), []]], [[rng.randint(0, 9)]]],))
        else:
            cases.append(([[s := rng.randint(0, 20), s + rng.randint(0, 6)] for _ in range(rng.randint(1, 5))],))
    return cases


def _extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else text


def run_python_tests(code: str, name: str, cases: list[tuple[Any, ...]], expected: list[Any], timeout: float = 10.0) -> bool:
    """Run untrusted generated code in an isolated, time-limited child process."""
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    harness = (
        code
        + "\n\nimport json, sys\n"
        + f"_cases = json.loads({json.dumps(json.dumps(cases))})\n"
        + f"print(json.dumps([{name}(*c) for c in _cases]))\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(harness)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", path], cwd=tmp, capture_output=True, text=True,
                timeout=timeout, env={"PATH": "/usr/bin:/bin", "HOME": tmp},
            )
        except subprocess.TimeoutExpired:
            return False
    if proc.returncode != 0:
        return False
    try:
        got = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False
    return bool(got == json.loads(json.dumps(expected)))


def _coding(rng: random.Random, i: int, split: str) -> Task:
    specs = _pool(CODE_SPECS, split)
    name, desc, fn, kind = specs[i % len(specs)]
    cases = _code_inputs(rng, kind)
    if kind == "nested":
        expected = [_flatten(c[0]) for c in cases]
    elif kind == "intervals":
        expected = [_merge([list(x) for x in c[0]]) for c in cases]
    elif kind == "pos":
        import math as _math_mod
        expected = [_math_mod.gcd(*c[0]) for c in cases]
    else:
        assert fn is not None
        expected = [fn(*c) for c in cases]
    q = f"Write a Python function named `{name}` that {desc}. Return only the code in one ```python block."
    return Task("coding", f"coding-{i}", [{"role": "user", "content": q}],
                lambda t, n=name, c=cases, e=expected: run_python_tests(_extract_code(t), n, c, e), 512)


HE_CAPITALS = [
    ("צרפת", ["פריז"]), ("איטליה", ["רומא"]), ("יפן", ["טוקיו"]), ("מצרים", ["קהיר"]), ("ספרד", ["מדריד"]),
    ("גרמניה", ["ברלין"]), ("רוסיה", ["מוסקבה"]), ("יוון", ["אתונה"]), ("קנדה", ["אוטווה", "אוטאווה"]),
    ("אוסטרליה", ["קנברה"]), ("ארגנטינה", ["בואנוס איירס"]), ("פורטוגל", ["ליסבון"]), ("נורווגיה", ["אוסלו"]),
    ("טורקיה", ["אנקרה"]), ("הודו", ["ניו דלהי", "דלהי"]), ("סין", ["בייג'ינג", "בייג׳ינג", "בייג'ין"]),
]
HE_NAMES = ["דנה", "יוסי", "מיכל", "אורי", "נועה", "עמית", "שירה", "רון"]


def _hebrew(rng: random.Random, i: int, split: str) -> Task:
    kind = i % 3
    if kind == 0:
        pool = _pool(HE_CAPITALS, split)
        country, answers = pool[rng.randrange(len(pool))]
        q = f"מהי עיר הבירה של {country}? ענה במילה או שתיים בעברית בלבד."
        return Task("hebrew", f"hebrew-{i}", [{"role": "user", "content": q}],
                    lambda t, a=answers: any(x in t for x in a), 64, reference=answers[0])
    if kind == 1:
        a, b = rng.sample(HE_NAMES, 2)
        x, y, z = rng.randint(3, 30), rng.randint(3, 30), rng.randint(1, 3)
        q = (f"ל{a} יש {x} ספרים ול{b} יש {y} ספרים. {a} נותנ/ת ל{b} {z} ספרים, ואז {b} קונה עוד {x} ספרים. "
             f"כמה ספרים יש עכשיו ל{b}?" + HE_ANSWER_SUFFIX)
        return Task("hebrew", f"hebrew-{i}", [{"role": "user", "content": q}], _num_check(y + z + x), 384, reference=str(y + z + x))
    name, city, age = rng.choice(HE_NAMES), rng.choice(["חיפה", "אילת", "ירושלים", "טבריה", "באר שבע"]), rng.randint(20, 70)
    passage = f"{name} גר/ה ב{city} כבר עשר שנים. בגיל {age} התחיל/ה ללמוד לנגן בפסנתר, ובכל בוקר הולך/ת לים."
    q = f"קרא את הטקסט וענה: באיזו עיר גר/ה {name}? ענה בשם העיר בלבד.\n\n{passage}"
    return Task("hebrew", f"hebrew-{i}", [{"role": "user", "content": q}], lambda t, c=city: c in t, 64, reference=city)


def _english(rng: random.Random, i: int, split: str) -> Task:
    name, job, pet, year = rng.choice(NAMES), rng.choice(["baker", "pilot", "nurse", "chemist", "carpenter"]), rng.choice(["parrot", "turtle", "beagle", "cat"]), rng.randint(1990, 2020)
    passage = (f"In {year}, {name} moved to a small coastal town and worked as a {job}. "
               f"On weekends {name} walked a {pet} along the harbor and kept a diary of the weather.")
    kind = i % 3
    question, ans = [("What is the person's job? One word.", job), ("What animal does the person walk? One word.", pet), ("In which year did the person move? Number only.", str(year))][kind]
    return Task("english", f"english-{i}", [{"role": "user", "content": f"{passage}\n\n{question}"}],
                lambda t, a=ans: a.lower() in t.lower(), 64, reference=ans)


def _instruction(rng: random.Random, i: int, split: str) -> Task:
    topic = rng.choice(_pool(["sleep", "rivers", "bicycles", "tea", "volcanoes", "libraries", "deserts", "pianos", "honey", "bridges", "owls", "glaciers"], split))
    kind = i % 5
    if kind == 0:
        n = rng.randint(3, 5)
        q = f"List exactly {n} facts about {topic} as bullet points starting with '- '. No other text."
        check: Callable[[str], bool] = lambda t, n=n: (lines := [ln for ln in t.strip().splitlines() if ln.strip()]) != [] and len(lines) == n and all(ln.lstrip().startswith("- ") for ln in lines)
    elif kind == 1:
        q = f"Write two sentences about {topic} using only lowercase letters."
        check = lambda t: t.strip() != "" and t == t.lower()
    elif kind == 2:
        word = rng.choice(_pool(["banana", "lantern", "orbit", "velvet", "copper", "meadow"], split))
        q = f"Write one sentence about {topic} that contains the word '{word}' exactly twice."
        check = lambda t, w=word: len(re.findall(rf"\b{w}\b", t.lower())) == 2
    elif kind == 3:
        q = f"Reply with a JSON object only, with keys \"topic\" (string, value '{topic}') and \"score\" (integer 1-10)."
        def check(t: str, tp: str = topic) -> bool:
            body = t.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                obj = json.loads(body)
            except ValueError:
                return False
            return isinstance(obj, dict) and set(obj) == {"topic", "score"} and obj["topic"] == tp and isinstance(obj["score"], int) and 1 <= obj["score"] <= 10
    else:
        n = rng.randint(8, 15)
        q = f"Describe {topic} in at most {n} words."
        check = lambda t, n=n: 0 < len(t.split()) <= n
    return Task("instruction_following", f"if-{i}", [{"role": "user", "content": q}], check, 256)


WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}
CALC_TOOL = {"type": "function", "function": {"name": "multiply", "description": "Multiply two integers exactly",
             "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for body in re.findall(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", text, re.S):
        try:
            obj = json.loads(body)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append({"name": obj["name"], "arguments": args})
    return calls


def _tool(rng: random.Random, i: int, split: str) -> Task:
    if i % 2 == 0:
        city = rng.choice(_pool(["Haifa", "Lisbon", "Osaka", "Nairobi", "Lima", "Oslo", "Quito", "Hanoi", "Perth"], split))
        q = f"What's the weather in {city} right now, in celsius?"
        check: Callable[[str], bool] = lambda t, c=city: any(x["name"] == "get_weather" and str(x["arguments"].get("city", "")).lower() == c.lower() for x in parse_tool_calls(t))
        return Task("tool_use", f"tool-{i}", [{"role": "user", "content": q}], check, 128, tools=[WEATHER_TOOL, CALC_TOOL])
    a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
    q = f"Use the multiply tool to compute {a} times {b}."
    check = lambda t, a=a, b=b: any(x["name"] == "multiply" and x["arguments"].get("a") in (a, b) and {x["arguments"].get("a"), x["arguments"].get("b")} == {a, b} for x in parse_tool_calls(t))
    return Task("tool_use", f"tool-{i}", [{"role": "user", "content": q}], check, 128, tools=[WEATHER_TOOL, CALC_TOOL])


FILLER = ["Can you suggest a name for a houseplant?", "Give me a one-line tip for focus.", "What rhymes with 'light'?",
          "Name a board game for four players.", "Suggest a healthy snack.", "Give one word that means 'happy'."]


def _long(rng: random.Random, i: int, split: str) -> Task:
    code = f"{rng.choice(['amber', 'cobalt', 'maple', 'quartz'])}-{rng.randint(100, 999)}"
    msgs: Messages = [{"role": "user", "content": f"Please remember my locker code: {code}. Just acknowledge."},
                      {"role": "assistant", "content": "Noted, I'll remember your locker code."}]
    for f in rng.sample(FILLER, 5) + rng.sample(FILLER, 3):
        msgs += [{"role": "user", "content": f}, {"role": "assistant", "content": "Sure — here is a short answer."}]
    msgs.append({"role": "user", "content": "What was my locker code? Reply with the code only."})
    return Task("long_conversation", f"long-{i}", msgs, lambda t, c=code: c in t, 32, reference=code)


FACTS = [
    ("What is the chemical symbol for gold?", ["au"]), ("How many planets are in the Solar System?", ["8", "eight"]),
    ("Who wrote 'Pride and Prejudice'?", ["austen"]), ("What is the largest ocean on Earth?", ["pacific"]),
    ("What is the boiling point of water at sea level in Celsius?", ["100"]), ("Which planet is known as the Red Planet?", ["mars"]),
    ("What is the square root of 144?", ["12"]), ("In which continent is Kenya?", ["africa"]),
    ("What gas do plants absorb for photosynthesis?", ["carbon dioxide", "co2"]), ("Who developed the theory of general relativity?", ["einstein"]),
    ("What is the capital of Canada?", ["ottawa"]), ("How many sides does a hexagon have?", ["6", "six"]),
    ("What is the hardest natural substance?", ["diamond"]), ("Which language has the most native speakers?", ["mandarin", "chinese"]),
    ("What is H2O commonly called?", ["water"]), ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]),
]


def _factual(rng: random.Random, i: int, split: str) -> Task:
    pool = _pool(FACTS, split)
    q, ans = pool[(i + rng.randrange(len(pool))) % len(pool)]
    return Task("factual", f"fact-{i}", [{"role": "user", "content": q + " Answer briefly."}],
                lambda t, a=ans: any(x in t.lower() for x in a), 64, reference=ans[0])


def _general(rng: random.Random, i: int, split: str) -> Task:
    kind = i % 2
    if kind == 0:
        item, price, qty = rng.choice(_pool(["lamp", "chair", "kettle", "desk", "blender", "mirror"], split)), rng.randint(20, 90), rng.randint(2, 6)
        q = (f"Write a short, polite email to a supplier ordering {qty} units of the {item} at ${price} each. "
             f"Mention the total cost.")
        return Task("general_assistance", f"gen-{i}", [{"role": "user", "content": q}],
                    lambda t, tot=price * qty, it=item: str(tot) in t.replace(",", "") and it in t.lower(), 384)
    city, n = rng.choice(_pool(["Rome", "Kyoto", "Haifa", "Prague", "Cusco", "Tbilisi"], split)), rng.randint(2, 4)
    q = f"Make a {n}-day travel plan for {city}. Use headings 'Day 1', 'Day 2', etc., one per day, nothing beyond Day {n}."
    return Task("general_assistance", f"gen-{i}", [{"role": "user", "content": q}],
                lambda t, n=n: all(f"day {d}" in t.lower() for d in range(1, n + 1)) and f"day {n + 1}" not in t.lower(), 1024)


GENERATORS: dict[str, Callable[[random.Random, int, str], Task]] = {
    "coding": _coding, "math": _math, "reasoning": _reasoning, "hebrew": _hebrew, "english": _english,
    "instruction_following": _instruction, "tool_use": _tool, "long_conversation": _long,
    "factual": _factual, "general_assistance": _general,
}


def build_tasks(split: str, per_category: int, categories: tuple[str, ...] = CATEGORIES) -> list[Task]:
    if split not in SPLIT_SEEDS:
        raise ValueError("split must be validation, test or train")
    tasks = []
    for c_index, category in enumerate(categories):
        rng = random.Random(SPLIT_SEEDS[split] + 1000 * c_index)
        tasks += [GENERATORS[category](rng, i, split) for i in range(per_category)]
    return tasks


def evaluate(generate: Callable[[Task], str], split: str = "validation", per_category: int = 8,
             categories: tuple[str, ...] = CATEGORIES, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Score a generator callable (task -> reply text) on procedural tasks."""
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for task in build_tasks(split, per_category, categories):
        t0 = time.perf_counter()
        reply = generate(task)
        ok = bool(task.check(reply))
        rows.append({"category": task.category, "id": task.task_id, "passed": ok,
                     "seconds": time.perf_counter() - t0, "reply": reply[-400:], "reference": task.reference})
        if log:
            log(f"{task.task_id} {'PASS' if ok else 'fail'}")
    scores = {c: sum(r["passed"] for r in rows if r["category"] == c) / max(1, sum(r["category"] == c for r in rows)) for c in categories}
    return {
        "split": split, "per_category": per_category, "scores": scores,
        "mean": sum(scores.values()) / len(scores),
        "meets_target": {c: scores[c] >= TARGET_THRESHOLDS[c] for c in categories},
        "target_profile": "provisional ModelForge frontier-class thresholds; not a measured GPT-5.6 Luna comparison",
        "seconds": time.perf_counter() - start, "rows": rows,
    }
