# Global Coding Guidelines

> Reference this file at the start of a Claude Code session with `@guidelines.md`.  
> These rules apply to every coding task unless explicitly overridden.

---

## 1. Session Flow

Every session follows this sequence. Do not skip steps.

| Step | Action |
|------|--------|
| **Ingest** | Accept the problem in any format: LaTeX, Markdown, plain text, or equations |
| **Clarify** | Ask clarifying questions one at a time; wait for answers before proceeding |
| **Architect** | Produce `ARCHITECTURE.md` (see Section 2) and lock it before writing any code |
| **Build** | Implement in phases (see Section 3), each gated by passing tests |
| **Iterate** | Refine, version, and clean up (see Section 8) |

---

## 2. Architecture (`ARCHITECTURE.md`)

Before any implementation, produce a `ARCHITECTURE.md` that covers:

- **Components** — what each module/class is responsible for
- **Data** — where it lives, what format it takes
- **Compute** — where the code runs (local / Lambda / Batch / SageMaker)
- **Phases** — breakdown matching Section 3
- **Hyperparameters** — schema matching Section 5

Review the file carefully:
- Spell out what each core component will look like
- Check for duplicated logic and conflicts between conditions
- Add enhancements only after the core is solid
- **Lock the architecture before writing any code**

---

## 3. Phased Development

Phases are determined by the problem. The default breakdown is:

| Phase | Focus | Gate |
|-------|-------|------|
| 1 — Data | Ingestion, parsing, validation | All data unit tests pass |
| 2 — Model | Core algorithm and business logic | All model unit tests pass |
| 3 — Deployment | AWS, Docker, environment | Integration tests pass |

Do not proceed to the next phase until the current phase's tests pass.

---

## 4. Code Quality

### Design
- **Object-oriented by default**: small, focused classes; simple public API; private implementation details
- **Reuse over duplication**: factor repeated behavior into methods or utilities; prefer composition
- **Single responsibility**: keep functions and methods small and single-purpose

### Performance
Default rule: **NumPy if cores don't matter; Numba if cores matter.**

- Use **NumPy broadcasting** for vectorized operations over arrays
- Use **Numba** (`@njit`, `prange`) when saturating CPU cores:

```python
from numba import njit, prange

@njit(parallel=True)
def my_function(arr):
    # Numba compiles on first call (warm start), then runs at C-like speed
    ...

@njit
def distance(a, b):
    for i in prange(len(a)):
        for j in range(len(b)):
            ...
```

- Use **multiprocessing or AWS** for embarrassingly parallel workloads

### Correctness & Safety
- Validate inputs at public API boundaries with informative error messages
- Use custom exceptions where they clarify the failure mode
- Never silently swallow exceptions

### Style
- Follow **PEP8**; use **type hints** throughout
- Provide **docstrings** for every module, class, and function
- State **time and space complexity** in the docstring of any non-trivial function
- Add short inline comments for non-obvious logic only
- Prefer well-known numerical methods (Newton-Raphson, gradient descent, bisection) over naive approaches where applicable

---

## 5. Hyperparameters

**Always** create a `config.yaml`. No magic numbers in code.

```yaml
model:
  learning_rate: 0.01
  max_iterations: 1000
data:
  input_path: data/input.csv
  output_path: data/output.csv
aws:
  region: us-east-1
  instance_type: ml.m5.xlarge
```

---

## 6. Visualization

Produce visual output at every opportunity. Plots are the fastest way to catch errors.

| Problem type | Recommended visual |
|---|---|
| Graph / routing | Network plot with nodes, edges, highlighted solution path |
| Optimization | Objective value vs. iteration with convergence marker |
| Scheduling | Gantt chart |
| Forecast / time-series | Actuals vs. predicted with confidence intervals |
| Tabular results | Formatted table via `tabulate` or `pandas` |
| Simulation | Progress animation via `matplotlib.animation` |
| Spatial / geographic | Map via `folium` or `geopandas` |

- Use `matplotlib` by default; `plotly` for interactive or animated output
- Always save figures to file in addition to displaying them

---

## 7. Environment & Deployment

### Local Setup

```bash
uv venv
source .venv/bin/activate
uv pip install numpy numba torch pandas matplotlib plotly tabulate
python my_code.py
```

Always source the virtual environment before running any Python code.

### AWS Compute — Choose One and Do Not Switch

| Service | When to use |
|---------|-------------|
| **Lambda** | Runtime < 15 min; massively parallel (1000s of jobs) |
| **Batch** | Runtime > 15 min; parallel jobs or single long job |
| **SageMaker** | Single large machine (EC2 24xlarge, GPU) |

Package with **Docker**. All deployment code lives in Phase 3.

---

## 8. Iterative Workflow

### During a session
1. Auto-approve safe commands (file creation, installs, test runs)
2. If a method is slow or wrong, stop and request a better approach
3. Never patch interactively — always produce a re-runnable script
4. Version improvements: `v2`, `v3`, `v4` as the solution evolves
5. When satisfied, consolidate and clean up

### Periodic checkpoints
- Update `DEVELOPING.md` with current status
- Commit to git (see Section 9)

### Optimization cycle
1. Re-read the full `.md` file fresh
2. Produce a written list of corrections
3. Apply all corrections in one pass
4. Repeat until stable

---

## 9. Git Workflow

### Commit format
```
<type>(<scope>): <short description>    ← 50 chars max, imperative, no period

<body explaining why, not what>         ← wrap at 72 chars
```

**Types:**

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Neither fix nor feature |
| `test` | Adding or correcting tests |
| `docs` | Documentation only |
| `chore` | Build, dependencies, tooling |
| `perf` | Performance improvement |

**Examples:**
```
feat(model): add XGBoost approximator for DP value function

Replaces tabular DP with XGBoost surrogate.
Reduces inference from O(n^3) to O(1) per query.
```
```
fix(data): handle missing demand values in CSV parser
```

### Branches
```
<type>/<short-description>
```
Examples: `feat/xgboost-approximator`, `fix/csv-parser`

### When to commit
- End of each phase, after tests pass
- Before any major refactor
- Never commit broken or untested code to `master`
- Open a PR into `master`; do not push directly

---

## 10. Deliverables

Every completed task produces:

| Artifact | Description |
|----------|-------------|
| `src/` | OOP source implementation |
| `config.yaml` | All hyperparameters |
| `tests/test_xyz.py` | Unit, runtime, and integration tests with results |
| `requirements.txt` | All dependencies |
| `ARCHITECTURE.md` | Components, data, compute, phases |
| `DEVELOPING.md` | Living log updated throughout the session |
| `README.md` | Purpose, inputs/outputs, assumptions, complexity, run/test commands, example usage |

---

## Notes

- If a task contradicts a guideline (e.g. "do not use classes"), follow the explicit instruction for that task only
- If a library restriction prevents adherence, explain and provide the best alternative
- LaTeX integration is valuable: keep formulations and code cross-referenced so they can be checked against each other