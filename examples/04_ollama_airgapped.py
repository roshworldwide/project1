"""Air-gapped evaluation with Ollama: zero bytes leave the machine.

Prereqs:
    ollama pull llama3.2          # once, while online
    ollama serve                  # then unplug the network if you like

Run: python examples/04_ollama_airgapped.py
"""

import sys

import httpx

from holdout import Case, Eval, run
from holdout.providers import Ollama
from holdout.scorers import ExactMatch

cases = [
    Case(input="Answer with one word: capital of Japan?", reference="Tokyo"),
    Case(input="Answer with one number: 7 * 8 = ?", reference="56"),
    Case(input="Answer with one word: color of a stop sign?", reference="Red"),
    Case(input="Answer with one word: opposite of 'cold'?", reference="Hot"),
    Case(input="Answer with one number: days in a week?", reference="7"),
    Case(input="Answer with one word: largest ocean?", reference="Pacific"),
]

ev = Eval("local-qa", cases, [ExactMatch()])
target = Ollama(
    "llama3.2",
    system="Answer with exactly the word or number requested. No punctuation.",
)

try:
    result = run(ev, target=target, seed=42)
except Exception as exc:  # no server running — explain instead of stack-tracing
    if isinstance(exc.__cause__, httpx.TransportError) or "ollama" in str(exc).lower():
        sys.exit("Ollama is not reachable on localhost:11434 — run 'ollama serve' first.")
    raise

print(result.summary())
print()
print("Everything above — generation, scoring, the bootstrap, the store — ran locally.")
