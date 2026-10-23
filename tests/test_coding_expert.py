"""Independent reference programs must pass every coding_expert checker (validates hidden tests)."""
import re

from modelforge import eval_suite as es

EXPR = '''
def NAME(s):
    toks = re.findall(r"\\d+|[-+*/()^]", s)
    pos = [0]
    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None
    def eat():
        pos[0] += 1
        return toks[pos[0] - 1]
    def expr():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if eat() == "+" else v - term()
        return v
    def term():
        v = un()
        while peek() in ("*", "/"):
            op = eat(); r = un()
            if op == "*":
                v = v * r
            else:
                q = abs(v) // abs(r)
                v = q if (v >= 0) == (r > 0) else -q
        return v
    def un():
        if peek() == "-":
            eat(); return -un()
        b = atom()
        if peek() == "^":
            eat(); return b ** un()
        return b
    def atom():
        t = eat()
        if t == "(":
            v = expr(); eat(); return v
        return int(t)
    return expr()
import re
'''
DECODE = '''
def NAME(s):
    stack, cur, num = [], "", 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch in "[{(<":
            stack.append((cur, num)); cur, num = "", 0
        elif ch in "]})>":
            p, k = stack.pop(); cur = p + cur * k
        else:
            cur += ch
    return cur
'''
DIJKSTRA = '''
import heapq
def NAME(n, edges, s, t):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        if not DIRECTED:
            adj[v].append((u, w))
    best = [None] * n
    h = [(0, s)]
    while h:
        d, u = heapq.heappop(h)
        if best[u] is not None:
            continue
        best[u] = d
        for v, w in adj[u]:
            if best[v] is None:
                heapq.heappush(h, (d + w, v))
    return -1 if best[t] is None else best[t]
'''
TOPO = '''
import heapq
def NAME(n, pairs):
    ind = [0] * n; adj = [[] for _ in range(n)]
    for a, b in pairs:
        adj[a].append(b); ind[b] += 1
    h = [SIGN * v for v in range(n) if ind[v] == 0]; heapq.heapify(h); out = []
    while h:
        u = SIGN * heapq.heappop(h); out.append(u)
        for v in adj[u]:
            ind[v] -= 1
            if ind[v] == 0:
                heapq.heappush(h, SIGN * v)
    return out if len(out) == n else None
'''
LRU = '''
class NAME:
    def __init__(self, capacity):
        self.c = capacity; self.d = {}
    def get(self, k):
        if k not in self.d:
            return MISSING
        v = self.d.pop(k); self.d[k] = v; return v
    def put(self, k, v):
        self.d.pop(k, None); self.d[k] = v
        if len(self.d) > self.c:
            del self.d[next(iter(self.d))]
'''
WB = '''
def NAME(s, words):
    w = set(words); dp = [0] * (len(s) + 1); dp[len(s)] = 1
    for i in range(len(s) - 1, -1, -1):
        dp[i] = sum(dp[j] for j in range(i + 1, len(s) + 1) if s[i:j] in w)
    return dp[0]
'''
KD = '''
def NAME(s):
    best = 0; last = {}; left = 0
    for r, ch in enumerate(s):
        last[ch] = r
        while len(last) > K:
            m = min(last, key=last.get); left = last.pop(m) + 1
        best = max(best, r - left + 1)
    return best
'''
MATCH = '''
from functools import lru_cache
def NAME(s, p):
    @lru_cache(None)
    def f(i, j):
        if j == len(p):
            return i == len(s)
        ok = i < len(s) and p[j] in (s[i], ".")
        q = p[j + 1] if j + 1 < len(p) else ""
        if q == "*":
            return f(i, j + 2) or (ok and f(i + 1, j))
        if q == "+":
            return ok and (f(i + 1, j) or f(i + 1, j + 2))
        return ok and f(i + 1, j + 1)
    return f(0, 0)
'''
TIMEMAP = '''
class NAME:
    def __init__(self):
        self.d = {}
    def set(self, k, v, t):
        self.d.setdefault(k, {})[t] = v
    def get(self, k, t):
        c = [x for x in self.d.get(k, {}) if x <= t]
        return self.d[k][max(c)] if c else ""
'''
QUEENS = '''
def NAME(n, blocked):
    b = set(map(tuple, blocked))
    def go(r, c, d1, d2):
        if r == n:
            return 1
        return sum(go(r + 1, c | 1 << x, d1 | {r - x}, d2 | {r + x}) for x in range(n)
                   if not c >> x & 1 and r - x not in d1 and r + x not in d2 and (r, x) not in b)
    return go(0, 0, frozenset(), frozenset())
'''
WINDOW = '''
def NAME(s, t):
    if not t:
        return ""
    from collections import Counter
    need = Counter(t); best = None
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            if not need - Counter(s[i:j]):
                if best is None or j - i < len(best):
                    best = s[i:j]
                break
    return best or ""
'''
GRID = '''
def NAME(g):
    r, c = len(g), len(g[0]); dp = [[0] * (c + 1) for _ in range(r + 1)]
    for x in range(r):
        for y in range(c):
            if g[x][y] == ".":
                dp[x + 1][y + 1] = 1 if x == y == 0 else dp[x][y + 1] + dp[x + 1][y]
    return dp[r][c] MODEXPR
'''


def solution(prompt: str) -> str:
    m = re.search(r"named `(\w+)`", prompt) or re.search(r"class `(\w+)\(", prompt)
    name = m.group(1)
    if "arithmetic expression" in prompt:
        src = EXPR
    elif "repeated k times" in prompt:
        src = DECODE
    elif "weighted edges" in prompt:
        src = DIJKSTRA.replace("DIRECTED", str("of directed" in prompt))
    elif "must come before" in prompt:
        src = TOPO.replace("SIGN", "-1" if "largest" in prompt else "1")
    elif "least recently used" in prompt:
        src = LRU.replace("MISSING", "-1" if "or -1 if absent" in prompt else "None")
    elif "dictionary words" in prompt:
        src = WB
    elif "distinct characters" in prompt:
        src = KD.replace("K", re.search(r"at most (\d)", prompt).group(1))
    elif "matches the whole" in prompt:
        src = MATCH
    elif "largest timestamp" in prompt:
        src = TIMEMAP
    elif "queens" in prompt:
        src = QUEENS
    elif "shortest contiguous substring" in prompt:
        src = WINDOW
    else:
        src = GRID.replace("MODEXPR", "% 1000000007" if "modulo" in prompt else "")
    return "```python\n" + src.replace("NAME", name) + "\n```"


def test_reference_programs_pass_all_expert_tasks() -> None:
    tasks = [t for sp in ("train", "validation", "test") for t in es.build_tasks(sp, 30, ("coding_expert",))]
    failed = [t.messages[0]["content"][:120] for t in tasks if not t.check(solution(t.messages[0]["content"]))]
    assert not failed, failed
    assert not tasks[0].check("```python\ndef nope(): pass\n```")
