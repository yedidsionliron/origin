# Global Coding Guidelines

These guidelines apply to any coding exercise I request. Reference this file at the
start of a Claude Code session with `@guidelines.md` to apply all rules below.

---

## Session Initialization (Architect Mode)

Before writing any code, spend time on architecture. A typical session should allocate
~90 minutes to planning before switching to implementation.

1. **Ingest the problem** in any of the following formats:
   - LaTeX file or equations
   - Markdown file
   - Plain paragraph
   
2. **Enter question mode**: ask clarifying questions one at a time. Answer them
   carefully before proceeding.

3. **Propose architecture** by producing a `ARCHITECTURE.md` file that specifies:
   - Core components and their responsibilities
   - Where the code will run (local / AWS Lambda / Batch / SageMaker)
   - Where data will live and what format it takes
   - Phase breakdown (see Phased Development below)
   - Hyperparameter schema (see Hyperparameters below)

4. **Review `ARCHITECTURE.md` carefully before approving:**
   - Catch logical mistakes and inconsistencies
   - Ask it to spell out what each core component will look like
   - Check for duplicated logic across conditions
   - Check for conflicts between conditions
   - Add bells and whistles only after the core is solid

5. **Lock the architecture** before writing any implementation code.

---

## Phased Development

Design and implement in three explicit phases, each with its own unit tests:

**Phase 1 — Data**
- Data ingestion, parsing, and validation
- Unit tests confirming correct loading and format

**Phase 2 — Model**
- Core algorithm and business logic
- Unit tests confirming correctness on known inputs

**Phase 3 — Deployment**
- AWS deployment, Docker, environment setup
- Integration tests confirming end-to-end correctness

Do not proceed to Phase 2 until Phase 1 tests pass. Same for Phase 3.

---

## Core Principles

1. **Understand before implementing**
   - Restate the problem in your own words before writing any code.
   - Confirm ambiguous assumptions explicitly.
   - Propose a 3-5 step plan and wait for approval if the task is non-trivial.

2. **Object oriented by default**
   - Prefer small, focused classes with clear responsibilities.
   - Expose a simple public API and keep implementation details private.

3. **Maximize reuse, minimize redundancy**
   - Factor repeated behavior into methods/utilities.
   - Prefer composition over duplication.

4. **Vectorization and performance**
   - Use NumPy broadcasting (vectorization) when CPU cores are not the bottleneck.
   - Use Numba (`@njit`, `prange`) when saturating CPU cores matters:
```python
     from numba import njit, prange

     @njit(parallel=True)
     def my_function(arr):
         # comments explaining logic
         ...

     @njit
     def distance(a, b):
         for i in range(len(a)):
             for j in range(len(b)):
                 # C-like speed operations
                 ...
```
   - Note: Numba compiles on first call (warm start), then runs at C-like speed.
   - Use multiprocessing or AWS for embarrassingly parallel workloads.
   - Default rule: **NumPy if cores don't matter; Numba if cores matter.**

5. **Visualize by default**
   - Create visual output whenever it aids understanding: graphs, plots, animations,
     and tables.
   - For network/routing problems: plot the graph and highlight the solution path.
   - For optimization problems: plot objective value vs. iteration.
   - For tabular data: render a formatted table via `tabulate` or `pandas`.
   - For time-series or forecasts: plot actuals vs. predicted with confidence intervals.
   - For simulations: progress animation using `matplotlib.animation`.
   - For spatial/geographic problems: map plot using `folium` or `geopandas`.
   - Use `matplotlib` by default; `plotly` for interactive or animated output.
   - Always save figures to file in addition to displaying them.
   - **Plots are the best way to catch errors --- produce them early and often.**

6. **Complexity awareness**
   - State time and space complexity in the docstring of any non-trivial function.
   - If a brute-force solution is implemented first, note its complexity and mark
     optimization opportunities: `# TODO: optimize - currently O(n^2)`

7. **Meaningful documentation**
   - Provide module/class/function docstrings.
   - Add short, focused comments for non-obvious logic.
   - Include type hints throughout.
   - Keep `DEVELOPING.md` updated periodically as the session progresses.

8. **Generic configuration**
   - Accept parameters via function args, a config dict/object, or environment
     variables.
   - Avoid hardcoded paths, credentials, or environment specifics.

9. **Style & quality**
   - Follow PEP8.
   - Keep functions/methods small and single-purpose.
   - Validate inputs at public API boundaries with informative error messages.
   - Use custom exceptions where they clarify the failure mode.
   - Never silently swallow exceptions.

10. **File organization**
    - Keep all operational code and related classes within a single source file when
      reasonable.
    - Tests live in a separate file: `tests/test_xyz.py`.

---

## Hyperparameters

- **Always** create a `config.yaml` and store all hyperparameters there.
- No magic numbers in code. Every tunable value lives in the YAML.
- Example structure:
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

## Environment Setup

Always use a virtual environment:
```bash
# Create and activate
uv venv
source .venv/bin/activate   # or: source ~/.venv/bin/activate

# Install dependencies
uv pip install numpy numba torch pandas matplotlib plotly tabulate

# Always source before running
source .venv/bin/activate
python my_code.py
```

Bake all packages into the virtual environment before running. Use `uv` as the
preferred pip-like tool.

---

## AWS Deployment Guide

Choose **one** compute target and do not switch mid-project:

| Service    | When to use                                              |
|------------|----------------------------------------------------------|
| Lambda     | Run time < 15 min; massively parallel (1000s of jobs)   |
| Batch      | Run time > 15 min; parallel jobs or single long job      |
| SageMaker  | Single large machine (EC2 24xlarge, GPU)                 |

Package with Docker. All deployment code lives in Phase 3.

---

## Interview Mode

When working under time constraints:
- Implement a correct brute-force solution first, then optimize.
- A working simple solution beats an incomplete elegant one.
- Mark optimization opportunities clearly with TODO comments.

---

## Iterative Improvement Workflow

As the code runs:
1. **Auto-approve** safe commands (file creation, installs, test runs).
2. If a method is slow or wrong, stop and ask for a better approach. Pass the
   relevant code in the console.
3. Never patch interactively --- always produce a script that can be re-run.
4. Version your improvements: `v2`, `v3`, `v4` as the solution evolves.
5. When satisfied, ask Claude to clean up and consolidate.

Periodically during the session:
- Push to git: `git add . && git commit -m "checkpoint" && git push`
- Ask Claude to update `DEVELOPING.md` with current status.

---

## Optimization Cycle

When refining an existing solution:
1. Ask Claude to read the entire `.md` file fresh.
2. Ask for a written list of corrections and improvements.
3. Ask it to tune the whole file in one pass.
4. Repeat until stable.

---

## Deliverables

Every task should produce:
- Source file(s) with OOP implementation
- `config.yaml` with all hyperparameters
- `tests/test_xyz.py` with unit, runtime, and integration tests; run and include
  results
- `requirements.txt` or `pyproject.toml`
- `ARCHITECTURE.md` describing components, data, and deployment
- `DEVELOPING.md` updated as the session progresses
- A Markdown summary containing:
  - Purpose
  - Inputs and outputs
  - Assumptions
  - Time and space complexity
  - Run / test commands
  - Example usage

---

## Visualization Reference

| Problem type          | Recommended visual                                      |
|-----------------------|---------------------------------------------------------|
| Graph / routing       | Network plot with nodes, edges, and highlighted path    |
| Optimization          | Objective value vs. iteration (with convergence marker) |
| Scheduling            | Gantt chart                                             |
| Forecast / time-series| Actuals vs. predicted with confidence intervals         |
| Tabular results       | Formatted table via `tabulate` or `pandas`              |
| Simulation            | Progress animation using `matplotlib.animation`         |
| Spatial / geographic  | Map plot using `folium` or `geopandas`                  |

---

## Git Commit Guidelines

### Commit Message Format

Use the **Conventional Commits** style:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

**Types:**

| Type       | When to use                                         |
|------------|-----------------------------------------------------|
| `feat`     | A new feature                                       |
| `fix`      | A bug fix                                           |
| `refactor` | Code change that is neither a fix nor a feature     |
| `test`     | Adding or correcting tests                          |
| `docs`     | Documentation only changes                          |
| `chore`    | Build process, dependency updates, tooling          |
| `perf`     | Performance improvement                             |

**Rules:**
- Subject line: **50 characters or fewer**, imperative mood ("Add X", not "Added X")
- No period at the end of the subject line
- Blank line between subject and body
- Body: wrap at 72 characters; explain *why*, not *what*

**Examples:**
```
feat(model): add XGBoost approximator for DP lookup

Replaces the brute-force DP table with an XGBoost surrogate.
Reduces inference time from O(n^3) to O(1) per query.

Closes #42
```

```
fix(data): handle missing demand values in CSV parser
```

```
refactor(experiments): move dp_lookup into Experiments-code/
```

### Branch Naming

```
<type>/<short-description>
```

Examples: `feat/xgboost-approximator`, `fix/csv-parser`, `refactor/dp-lookup`

### When to Commit

- Commit at the end of each **phase** (Phase 1, 2, 3).
- Commit after each passing test suite.
- Commit before any major refactor (safe checkpoint).
- Never commit broken or untested code to `master`.

### Workflow

```bash
git add <specific-files>          # prefer specific files over git add .
git commit -m "type(scope): msg"
git push origin <branch>
```

- Open a PR into `master`; do not push directly to `master`.
- Squash fixup commits before merging.

---

## Notes

- If a task explicitly contradicts any guideline (e.g. "Do not use classes"), follow
  the explicit instruction for that task only.
- If a library or environment restriction prevents full adherence, explain the
  limitation and provide the best alternative.
- LaTeX integration is valuable: keep formulation and code cross-referenced so they
  can be checked against each other.