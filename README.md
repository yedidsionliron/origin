# XGBoost DP Lookup Table

This project implements a finite-horizon dynamic programming approximator
using XGBoost regressors.  The value function is trained backwards in time
on a set of state samples, replacing a traditional lookup table.

## Structure

- `dp_lookup.py`: main implementation (single-file operatonal code)
- `tests/test_dp_lookup.py`: simple unit tests
- `requirements.txt`: dependencies

## Quickstart

```bash
cd "C:\Users\Lirony\Downloads\Python codes"
pip install -r requirements.txt
python dp_lookup.py          # run demo
pytest -q                   # run tests
```
