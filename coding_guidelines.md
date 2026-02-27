# Global Coding Guidelines

These guidelines apply to any coding exercise I request. When I ask you to "Apply Global Coding Guidelines" (or any of the trigger phrases below), follow the rules in this document.

---

## Trigger phrases
- "Apply Global Coding Guidelines"
- "Follow my coding rules"
- "Use my standard template"

Use these phrases at the start of a prompt to request enforcement of these rules.

---

## Core Principles
1. Object oriented by default
   - Prefer small, focused classes with clear responsibilities.
   - Expose a simple public API and keep implementation details private.

2. Maximize reuse, minimize redundancy
   - Factor repeated behavior into methods/utilities.
   - Prefer composition over duplication.

3. Vectorization first
   - Use NumPy/Pandas or language-specific bulk operations; avoid element-wise loops where feasible.
   - Only use loops when vectorized approaches are infeasible or less clear.

4. Meaningful documentation
   - Provide module/class/function docstrings.
   - Include a README snippet explaining assumptions, usage, and run commands.
   - Add short, focused comments for non-obvious logic.

5. Generic configuration
   - Accept parameters via function args, a config dict/object, or environment variables.
   - Avoid hardcoded paths, credentials, or environment specifics.

6. Deliverables and tests
   - Produce source files, a `requirements.txt` (or `pyproject.toml`), a minimal runnable example, and at least one small unit test or basic verification snippet.

7. Style & quality
   - Use type hints where appropriate.
   - Follow PEP8 or common idioms for the language.
   - Keep functions/methods small and single-purpose.

8. File organization
   - Keep all operational code and related classes within a single source file when reasonable.
   - Tests should live in a separate file (e.g. `tests/test_xyz.py`) so they can be executed independently.
   - When executing a task, create code, then add a test file, run the tests, and include results in your response.

---

## Documentation / Markdown
- Each task should include a short Markdown summary with:
  - Purpose
  - Inputs and outputs
  - Assumptions
  - Run / test commands
- Include example usage showing how to call the main class or function.

---

## Configuration for the Assistant
When you receive a new coding request that includes the trigger phrase, apply these steps by default:
1. Propose a small plan (3–5 steps) and confirm assumptions if ambiguous.
2. Implement an object-oriented solution prioritizing reuse and vectorization.
3. Add docstrings, an example runner, and `requirements.txt` if needed.
4. Add a short test or usage snippet and a README section.
5. Keep changes minimal and focused; ask before making broad repo-wide edits.

---

## Example (what I expect)
- Request: "Apply Global Coding Guidelines: implement X"
- Outcome:
  - `src/` with an OOP implementation
  - `tests/test_x.py` with a basic test
  - `requirements.txt` listing dependencies
  - `README.md` snippet with run/test commands

---

## Notes
- If a task explicitly contradicts any guideline (e.g., "Do not use classes"), follow the explicit instruction for that task only.
- If a library or environment restriction prevents full adherence, explain the limitation and provide the best alternative.

---

Saved at: `C:/Users/Lirony/Downloads/Python codes/coding_guidelines.md`
