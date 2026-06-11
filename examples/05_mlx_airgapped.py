"""Air-gapped evaluation on Apple silicon with MLX: in-process inference.

Prereqs (Apple silicon only):
    pip install 'holdout[mlx]'
    # First run downloads the model; every run after that is fully offline.

Run: python examples/05_mlx_airgapped.py
"""

import sys

from holdout import Case, Eval, run
from holdout.exceptions import MissingDependencyError
from holdout.scorers import ExactMatch

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

cases = [
    Case(input="Answer with one word: capital of Italy?", reference="Rome"),
    Case(input="Answer with one number: 6 * 9 = ?", reference="54"),
    Case(input="Answer with one word: frozen water is called?", reference="Ice"),
    Case(input="Answer with one number: legs on a spider?", reference="8"),
]

ev = Eval("mlx-qa", cases, [ExactMatch()])

try:
    from holdout.providers import MLX

    target = MLX(MODEL, system="Answer with exactly the word or number requested.")
except MissingDependencyError as exc:
    sys.exit(str(exc))

# MLX inference is in-process and single-stream; keep concurrency at 1.
result = run(ev, target=target, seed=42, max_concurrency=1)
print(result.summary())
print()
print("Model weights, inference, scoring, statistics: all on this machine. 0 bytes out.")
