# Coding Guidelines — Light
> For small local tasks and timed interviews (≤ 30 min).

---

## Session Flow

| Step | Action | Time Budget |
|------|--------|-------------|
| **Clarify** | Ask at most ONE clarifying question | 2 min |
| **Plan** | State approach in 3–5 bullet points — no doc needed | 3 min |
| **Build** | Write working code in one pass | 20 min |
| **Test & Fix** | Run, verify output, patch if needed | 5 min |

---

## Code Quality (non-negotiable)

- **Type hints** on every function signature
- **One docstring** per function (one-liner is fine)
- **No magic numbers** — use named constants at the top of the file
- **No silent exceptions** — let errors surface
- Input validation only at the public entry point

---

## Performance Defaults

- Plain Python / list comprehensions first
- **NumPy** if working with arrays or math
- **Numba `@njit`** only if NumPy is provably the bottleneck

---

## Structure
```
solution.py       ← single file is fine for interviews
config.py         ← constants and parameters (skip if < 3 params)
test_solution.py  ← 3–5 focused tests
```

No AWS, no Docker, no virtual environments unless asked.

---

## Visualization

Add a quick plot if it helps verify correctness — one `matplotlib` call saved to `output.png`.

---

## Git (if time allows)
```bash
git add .
git commit -m "feat: <what it does>"
```

One commit per working solution. No branches needed.

---

## Deliverables Checklist

- [ ] `solution.py` — typed, docstrings, no magic numbers
- [ ] `test_solution.py` — tests pass
- [ ] `README.md` — 5 lines: purpose, inputs/outputs, how to run

---

## Interview Tips

1. **Talk through your plan before coding** — interviewers reward process
2. **Name constants clearly** — `MAX_ITERATIONS = 1000` beats `1000`
3. **Write the happy path first**, then edge cases
4. **Run early, run often** — don't wait until the end to test
5. If stuck, state your assumption out loud and keep moving