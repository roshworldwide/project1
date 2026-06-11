# Examples

| example | needs | shows |
|---|---|---|
| [01_quickstart.py](01_quickstart.py) | nothing (offline) | define, run, read estimates with CIs |
| [02_regression_gate.py](02_regression_gate.py) | nothing (offline) | noise stays green, real regressions fail with evidence |
| [03_pytest_suite/](03_pytest_suite/test_prompts.py) | nothing (offline) | evals as plain pytest tests: gates, power, leakage, @llm_eval |
| [04_ollama_airgapped.py](04_ollama_airgapped.py) | local Ollama | a full eval with 0 bytes leaving the machine |
| [05_mlx_airgapped.py](05_mlx_airgapped.py) | Apple silicon + `holdout[mlx]` | in-process inference, fully offline |
| [ci/](ci/) | nothing (runs in GitHub Actions) | the dogfood: this repo gating itself with its own action |

Run any of them from the repo root, e.g.:

```console
python examples/01_quickstart.py
pytest examples/03_pytest_suite/ -v
```
