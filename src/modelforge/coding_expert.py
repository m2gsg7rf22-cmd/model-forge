# ruff: noqa: E731
# mypy: disable-error-code="misc,assignment,arg-type,no-any-return"
"""``coding_expert`` tier: multi-step algorithmic and stateful-class programming tasks.

The v2 ``coding_hard`` tier saturated (~0.97), so this tier asks for parsers,
graph algorithms, dynamic programming with edge cases and small stateful
classes. Each prompt carries one worked example; the checker runs 8-12 hidden
cases (including edge cases) in an isolated child process. Prompts are unique
per seed and partitioned by the suite's hash split like every other category.
"""

from __future__ import annotations

import ast
import heapq
import json
import os
import random
import subprocess
import sys
import tempfile
from functools import lru_cache
from typing import Any, Callable

from .capability_eval import Task, _extract_code, run_python_tests


def run_class_tests(code: str, cls: str, scenarios: list[list[Any]], expected: list[Any], timeout: float = 10.0) -> bool:
    """Each scenario is [ctor_args, [[method, args], ...]]; compares the list of method results."""
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    harness = (code + "\n\nimport json\n"
               + f"_sc = json.loads({json.dumps(json.dumps(scenarios))})\n"
               + "_out = []\n"
               + "for _ctor, _ops in _sc:\n"
               + f"    _o = {cls}(*_ctor)\n"
               + "    _out.append([getattr(_o, _m)(*_a) for _m, _a in _ops])\n"
               + "print(json.dumps(_out))\n")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "candidate.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(harness)
        try:
            proc = subprocess.run([sys.executable, "-I", "-S", path], cwd=tmp, capture_output=True, text=True,
                                  timeout=timeout, env={"PATH": "/usr/bin:/bin", "HOME": tmp})
        except subprocess.TimeoutExpired:
            return False
    if proc.returncode != 0:
        return False
    try:
        got = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return False
    return bool(got == json.loads(json.dumps(expected)))


# ---------------------------------------------------------------- reference solutions

def _eval_expr(s: str, power: bool) -> int:
    toks: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            toks.append(s[i:j])
            i = j
        elif ch in "+-*/()^":
            toks.append(ch)
            i += 1
        else:
            i += 1
    pos = 0

    def peek() -> str | None:
        return toks[pos] if pos < len(toks) else None

    def eat() -> str:
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def expr() -> int:
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if eat() == "+" else v - term()
        return v

    def term() -> int:
        v = unary()
        while peek() in ("*", "/"):
            op, r = eat(), unary()
            v = v * r if op == "*" else (abs(v) // abs(r)) * (1 if (v < 0) == (r < 0) else -1)
        return v

    def unary() -> int:
        if peek() == "-":
            eat()
            return -unary()
        return pw()

    def pw() -> int:
        b = atom()
        if power and peek() == "^":
            eat()
            return b ** unary()
        return b

    def atom() -> int:
        t = eat()
        if t == "(":
            v = expr()
            eat()
            return v
        return int(t)

    return expr()


def _rand_expr(rng: random.Random, depth: int, power: bool) -> str:
    if depth == 0 or rng.random() < 0.3:
        return str(rng.randint(0, 9)) if rng.random() < 0.8 else "-" + str(rng.randint(1, 9))
    op = rng.choice(["+", "-", "*", "/"] + (["^"] if power else []))
    left = _rand_expr(rng, depth - 1, power)
    if op == "^":
        right = str(rng.randint(0, 3))
    elif op == "/":
        right = str(rng.randint(1, 9))
    else:
        right = _rand_expr(rng, depth - 1, power)
    text = f"{left} {op} {right}"
    return f"({text})" if rng.random() < 0.5 else text


def _decode(s: str, o: str, c: str) -> str:
    stack: list[tuple[str, int]] = []
    cur, num = "", 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == o:
            stack.append((cur, num))
            cur, num = "", 0
        elif ch == c:
            prev, k = stack.pop()
            cur = prev + cur * k
        else:
            cur += ch
    return cur


def _rand_encoded(rng: random.Random, depth: int, o: str, c: str) -> str:
    parts = []
    for _ in range(rng.randint(1, 3)):
        if depth and rng.random() < 0.5:
            parts.append(f"{rng.randint(1, 4)}{o}{_rand_encoded(rng, depth - 1, o, c)}{c}")
        else:
            parts.append("".join(rng.choice("abxyz") for _ in range(rng.randint(1, 3))))
    return "".join(parts)


def _dijkstra(n: int, edges: list[list[int]], src: int, dst: int, directed: bool) -> int:
    adj: dict[int, list[tuple[int, int]]] = {v: [] for v in range(n)}
    for u, v, w in edges:
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))
    dist = {src: 0}
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            return d
        if d > dist.get(u, 10**18):
            continue
        for v, w in adj[u]:
            if d + w < dist.get(v, 10**18):
                dist[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return -1


def _topo(n: int, edges: list[list[int]], largest: bool) -> list[int] | None:
    indeg = [0] * n
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    heap = [(-v if largest else v) for v in range(n) if indeg[v] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        u = abs(heapq.heappop(heap))
        out.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, -v if largest else v)
    return out if len(out) == n else None


def _lru(capacity: int, ops: list[list[Any]], missing: Any) -> list[Any]:
    from collections import OrderedDict

    store: OrderedDict[Any, Any] = OrderedDict()
    out = []
    for m, a in ops:
        if m == "get":
            if a[0] in store:
                store.move_to_end(a[0])
                out.append(store[a[0]])
            else:
                out.append(missing)
        else:
            store[a[0]] = a[1]
            store.move_to_end(a[0])
            if len(store) > capacity:
                store.popitem(last=False)
            out.append(None)
    return out


def _word_break_ways(s: str, words: list[str]) -> int:
    ws = set(words)

    @lru_cache(maxsize=None)
    def go(i: int) -> int:
        if i == len(s):
            return 1
        return sum(go(j) for j in range(i + 1, len(s) + 1) if s[i:j] in ws)

    return go(0)


def _at_most_k(s: str, k: int) -> int:
    from collections import Counter

    cnt: Counter[str] = Counter()
    best = left = 0
    for right, ch in enumerate(s):
        cnt[ch] += 1
        while len(cnt) > k:
            cnt[s[left]] -= 1
            if not cnt[s[left]]:
                del cnt[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best


def _match(s: str, p: str) -> bool:
    @lru_cache(maxsize=None)
    def go(i: int, j: int) -> bool:
        if j == len(p):
            return i == len(s)
        first = i < len(s) and p[j] in (s[i], ".")
        if j + 1 < len(p) and p[j + 1] == "*":
            return go(i, j + 2) or (first and go(i + 1, j))
        if j + 1 < len(p) and p[j + 1] == "+":
            return first and (go(i + 1, j + 2) or go(i + 1, j))
        return first and go(i + 1, j + 1)

    return go(0, 0)


def _queens(n: int, blocked: list[list[int]]) -> int:
    bl = {tuple(b) for b in blocked}

    def go(r: int, cols: frozenset[int], d1: frozenset[int], d2: frozenset[int]) -> int:
        if r == n:
            return 1
        return sum(go(r + 1, cols | {c}, d1 | {r - c}, d2 | {r + c}) for c in range(n)
                   if c not in cols and r - c not in d1 and r + c not in d2 and (r, c) not in bl)

    return go(0, frozenset(), frozenset(), frozenset())


def _min_window(s: str, t: str) -> str:
    from collections import Counter

    if not t:
        return ""
    need, have = Counter(t), Counter[str]()
    missing, left, best = len(t), 0, (10**9, 0, 0)
    for right, ch in enumerate(s):
        if need[ch] > have[ch]:
            missing -= 1
        have[ch] += 1
        while missing == 0:
            if right - left + 1 < best[0]:
                best = (right - left + 1, left, right + 1)
            have[s[left]] -= 1
            if have[s[left]] < need[s[left]]:
                missing += 1
            left += 1
    return s[best[1]:best[2]] if best[0] < 10**9 else ""


def _rand_graph(rng: random.Random, n: int, m: int, wmax: int) -> list[list[int]]:
    return [[rng.randrange(n), rng.randrange(n), rng.randint(1, wmax)] for _ in range(m)]


def _fmt(name: str, args: tuple[Any, ...], result: Any) -> str:
    return f"{name}({', '.join(json.dumps(a) for a in args)}) == {json.dumps(result)}"


# ---------------------------------------------------------------- generator

def coding_expert(rng: random.Random, i: int, split: str) -> Task:
    k = rng.randrange(12)
    fn: Callable[..., Any]
    cases: list[tuple[Any, ...]]
    cls_spec: tuple[str, list[list[Any]], list[Any]] | None = None
    if k == 0:
        power = rng.random() < 0.5
        name = rng.choice(["evaluate", "calc", "eval_expression"])
        desc = ("takes a string arithmetic expression over non-negative integer literals with +, -, *, /"
                + (", ^ (exponent, right-associative, binds tighter than unary minus on its left operand only as in -2^2 == -4)" if power else "")
                + ", parentheses, unary minus and arbitrary spaces, and returns its integer value. Standard precedence; "
                "/ is integer division truncating toward zero (like int(a / b)). Do not use eval or exec")
        fn = lambda e, p=power: _eval_expr(e, p)
        cases = [(_rand_expr(rng, 3, power),) for _ in range(10)] + [("-(3 - 10) / 2",), ("((7))",)]
    elif k == 1:
        o, c = rng.choice([("[", "]"), ("{", "}"), ("(", ")"), ("<", ">")])
        name = rng.choice(["decode", "expand", "decode_string"])
        desc = (f"decodes strings where k{o}text{c} means text repeated k times (k is a positive integer, possibly multi-digit); "
                f"groups can nest, e.g. 2{o}a3{o}b{c}{c} decodes to abbbabbb")
        fn = lambda s, oo=o, cc=c: _decode(s, oo, cc)
        cases = [(_rand_encoded(rng, 2, o, c),) for _ in range(9)] + [("",), (f"12{o}z{c}",)]
    elif k == 2:
        directed = rng.random() < 0.5
        name = rng.choice(["shortest_path", "min_cost_path", "cheapest_route"])
        desc = (f"takes n (nodes 0..n-1), a list of {'directed' if directed else 'undirected'} weighted edges [u, v, w] (w > 0, "
                "parallel edges and self-loops possible), a source and a target, and returns the minimum total weight "
                "from source to target, 0 if they are equal, or -1 if unreachable")
        fn = lambda n, e, s, t, d=directed: _dijkstra(n, e, s, t, d)
        cases = []
        for _ in range(9):
            n = rng.randint(2, 8)
            cases.append((n, _rand_graph(rng, n, rng.randint(0, 12), 20), rng.randrange(n), rng.randrange(n)))
        cases.append((3, [], 0, 0))
    elif k == 3:
        largest = rng.random() < 0.5
        name = rng.choice(["topo_order", "task_order", "schedule_order"])
        desc = (f"takes n (tasks 0..n-1) and a list of [a, b] pairs meaning a must come before b, and returns the "
                f"{'lexicographically largest' if largest else 'lexicographically smallest'} valid ordering as a list, or None if "
                "the constraints contain a cycle")
        fn = lambda n, e, lg=largest: _topo(n, e, lg)
        cases = []
        for _ in range(9):
            n = rng.randint(1, 7)
            cases.append((n, [[rng.randrange(n), rng.randrange(n)] for _ in range(rng.randint(0, 8))]))
        cases.append((3, [[0, 1], [1, 2], [2, 0]]))
    elif k == 4:
        missing = rng.choice([-1, None])
        cls = rng.choice(["LRUCache", "Cache", "RecentCache"])
        name = cls
        desc = (f"a class `{cls}(capacity)` with methods get(key) returning the value or {json.dumps(missing)} if absent, and "
                "put(key, value) returning None; both count as a use. When an insert exceeds capacity, evict the least "
                "recently used key. Updating an existing key changes its value and makes it most recent")
        scenarios, expected = [], []
        for _ in range(6):
            cap = rng.randint(1, 3)
            ops = [["put", [rng.randint(1, 5), rng.randint(0, 99)]] if rng.random() < 0.5 else ["get", [rng.randint(1, 5)]]
                   for _ in range(rng.randint(3, 12))]
            scenarios.append([[cap], ops])
            expected.append(_lru(cap, ops, missing))
        cls_spec = (cls, scenarios, expected)
        example = f"c = {cls}(1); c.put(1, 5); c.put(2, 6); c.get(1) -> {json.dumps(missing)}; c.get(2) -> 6"
    elif k == 5:
        name = rng.choice(["count_segmentations", "word_break_ways", "split_ways"])
        desc = ("takes a string s and a list of dictionary words and returns the number of distinct ways to split s entirely "
                "into a sequence of dictionary words (words may repeat); an empty s has exactly one way")
        fn = _word_break_ways
        cases = []
        for _ in range(9):
            words = sorted({"".join(rng.choice("ab") for _ in range(rng.randint(1, 3))) for _ in range(rng.randint(1, 5))})
            s = "".join(rng.choice(words) for _ in range(rng.randint(0, 5))) if rng.random() < 0.8 else "abba" * rng.randint(1, 3)
            cases.append((s, words))
        cases.append(("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", ["a", "aa", "aaa"]))
    elif k == 6:
        kk = rng.randint(1, 3)
        name = rng.choice(["longest_k_distinct", "longest_window", "max_k_distinct_len"])
        desc = f"takes a string s and returns the length of the longest contiguous substring containing at most {kk} distinct characters"
        fn = lambda s, q=kk: _at_most_k(s, q)
        cases = [("".join(rng.choice("abcd") for _ in range(rng.randint(0, 14))),) for _ in range(10)]
        cases.append(("x" * 2000 + "yz" * 1000,))
    elif k == 7:
        plus = rng.random() < 0.5
        name = rng.choice(["is_match", "pattern_match", "full_match"])
        desc = ("takes a string s and a pattern p and returns True if p matches the whole of s, where '.' matches any single "
                "character, 'x*' matches zero or more of the preceding element"
                + (" and 'x+' matches one or more of the preceding element" if plus else "")
                + "; do not use the re module")
        fn = lambda s, p: _match(s, p)
        cases = []
        for _ in range(10):
            s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 6)))
            p = ""
            for _ in range(rng.randint(1, 4)):
                p += rng.choice("ab.") + (rng.choice(["*", "+"] if plus else ["*"]) if rng.random() < 0.5 else "")
            cases.append((s, p))
        cases.append(("a" * 25 + "b", "a*" * 8 + "c"))
    elif k == 8:
        cls = rng.choice(["TimeMap", "VersionedStore", "HistoryKV"])
        name = cls
        desc = (f"a class `{cls}()` with set(key, value, timestamp) returning None and get(key, timestamp) returning the value "
                "set for key with the largest timestamp <= the given timestamp, or \"\" if there is none. Timestamps for a key "
                "may arrive in any order; a later set with an equal timestamp overwrites the earlier one")
        scenarios, expected = [], []
        for _ in range(6):
            kv_ops: list[list[Any]] = []
            for _ in range(rng.randint(3, 12)):
                key = rng.choice("pq")
                if rng.random() < 0.5:
                    kv_ops.append(["set", [key, rng.choice(["v1", "v2", "v3", "v4"]), rng.randint(1, 9)]])
                else:
                    kv_ops.append(["get", [key, rng.randint(0, 10)]])
            scenarios.append([[], kv_ops])
            data: dict[str, dict[int, str]] = {}
            exp: list[Any] = []
            for m, a in kv_ops:
                if m == "set":
                    data.setdefault(a[0], {})[a[2]] = a[1]
                    exp.append(None)
                else:
                    ts = [t for t in data.get(a[0], {}) if t <= a[1]]
                    exp.append(data[a[0]][max(ts)] if ts else "")
            expected.append(exp)
        cls_spec = (cls, scenarios, expected)
        example = f"m = {cls}(); m.set('a', 'x', 5); m.get('a', 4) -> ''; m.get('a', 7) -> 'x'"
    elif k == 9:
        name = rng.choice(["count_queens", "queen_placements", "n_queens_blocked"])
        desc = ("takes n and a list of blocked cells [row, col] and returns the number of ways to place n non-attacking "
                "queens on an n x n board with no queen on a blocked cell")
        fn = _queens
        cases = []
        for _ in range(7):
            m = rng.randint(4, 7)
            cases.append((m, [[rng.randrange(m), rng.randrange(m)] for _ in range(rng.randint(0, 3))]))
        cases += [(1, []), (2, []), (8, [])]
    elif k == 10:
        name = rng.choice(["min_window", "smallest_cover", "min_covering_substring"])
        desc = ("takes strings s and t and returns the shortest contiguous substring of s that contains every character of t "
                "with at least its multiplicity in t; if several have the same length return the leftmost; return \"\" if "
                "none exists or t is empty")
        fn = _min_window
        cases = []
        for _ in range(10):
            s = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 14)))
            t = "".join(rng.choice("abc") for _ in range(rng.randint(0, 3)))
            cases.append((s, t))
    else:
        mod = rng.choice([None, 1_000_000_007])
        name = rng.choice(["count_paths_grid", "grid_paths", "obstacle_paths"])
        desc = ("takes a grid (list of strings, '.' free and '#' blocked) and returns the number of paths from the top-left to "
                "the bottom-right cell moving only right or down through free cells (0 if either end is blocked)"
                + (", modulo 1000000007" if mod else ""))

        def paths(g: list[str], md: int | None = mod) -> int:
            r, c = len(g), len(g[0])
            dp = [[0] * c for _ in range(r)]
            for x in range(r):
                for y in range(c):
                    if g[x][y] == "#":
                        continue
                    dp[x][y] = 1 if (x, y) == (0, 0) else (dp[x - 1][y] if x else 0) + (dp[x][y - 1] if y else 0)
                    if md:
                        dp[x][y] %= md
            return dp[r - 1][c - 1]

        fn = paths
        cases = []
        for _ in range(8):
            rows, cols = rng.randint(1, 6), rng.randint(1, 6)
            cases.append((["".join("#" if rng.random() < 0.2 else "." for _ in range(cols)) for _ in range(rows)],))
        cases.append(([("." * 18)] * 18,))
    if cls_spec is None:
        exp_vals = [fn(*c) for c in cases]
        example = _fmt(name, cases[0], exp_vals[0])
        check = lambda t, n=name, cs=cases, e=exp_vals: run_python_tests(_extract_code(t), n, cs, e)
        what = f"a Python function named `{name}` that {desc}"
    else:
        cls, scenarios, expected = cls_spec
        check = lambda t, n=cls, sc=scenarios, e=expected: run_class_tests(_extract_code(t), n, sc, e)
        what = f"Python code defining {desc}"
    q = f"Write {what}.\nExample: {example}\nHandle edge cases carefully. Return only the code in one ```python block."
    return Task("coding_expert", f"ce{i}", [{"role": "user", "content": q}], check, 1536)
