# Leakage and holdout discipline

An eval can lie three ways without a single bug: its cases can hide inside the
prompt (contamination), its cases can repeat each other (effective-n
inflation), and its verdicts can decay through reuse (overfitting to the eval).
`holdout.leakage` ships one check for each.

## Contamination: eval data hiding in the prompt

The quiet way an eval lies: a case input — or worse, its reference answer —
appears verbatim or near-verbatim in the system prompt or few-shot examples,
and the model "solves" it by recall. The score measures memorization of your
own prompt, not capability.

`check_contamination(ev, corpus)` compares every case's input and reference
against a corpus of prompt text, with no model and no network. Two passes per
field:

1. **Exact substring, at word boundaries.** The field's word sequence appears
   verbatim in a corpus text (score 1.0). Matching is on casefolded,
   punctuation-stripped tokens joined with explicit boundaries, so a one-letter
   reference cannot "match" inside an unrelated word.
2. **Word n-gram containment.** At least `threshold` (default 0.5) of the
   field's word `ngram_size`-grams (default 5) appear in a corpus text. N-gram
   overlap is the standard embedding-free contamination check — the GPT-3
   evaluation used 13-gram dedup against its training data (Brown et al. 2020,
   *Language Models are Few-Shot Learners*, app. C); eval prompts are far
   shorter than training corpora, so holdout defaults to 5-grams and pairs the
   check with exact-substring matching. Containment (not Jaccard) is the right
   asymmetric measure for "is this small case inside that big prompt" — it is
   not diluted by the corpus's size. Fields shorter than `min_tokens` words
   skip this pass (a two-word answer matching a prompt is noise, not leakage)
   but are still checked for exact presence.

```python
from holdout.leakage import check_contamination

report = check_contamination(ev, SYSTEM_PROMPT, ngram_size=5, threshold=0.5)
if not report.clean:
    print(report.summary())
```

```text
contamination check (exact-substring + 5-gram containment >= 0.5): 2/4 case(s) flagged
  reset-pw.input: exact-substring (score=1.000) — appears verbatim in corpus text #0
  reset-pw.reference: exact-substring (score=1.000) — appears verbatim in corpus text #0
```

### The embedding pass

N-grams miss paraphrase: "How do I change my password?" in the prompt and
"What's the way to update my password?" in the eval share no 5-grams.
`check_contamination_embeddings()` embeds every case field and every corpus
text and flags cosine similarity at or above a threshold (default 0.9). Use a
local backend to keep the audit air-gapped:

```python
from holdout.leakage import check_contamination_embeddings
from holdout.providers import OllamaEmbeddings

report = await check_contamination_embeddings(
    ev, SYSTEM_PROMPT, OllamaEmbeddings("nomic-embed-text"), threshold=0.9
)
```

## Near-duplicates and effective n

Near-copies in a test set quietly inflate the effective sample size: 200 cases
that are really 150 unique problems do not buy the confidence interval of
n=200, and every test in the statistics engine assumes (conditionally)
independent cases. `find_near_duplicates(ev)` compares every pair of case
inputs by word n-gram Jaccard similarity (3-grams by default — inputs are
short) plus an exact check on normalized text, and returns pairs at or above
the threshold (default 0.8), most similar first:

```python
from holdout.leakage import find_near_duplicates

for pair in find_near_duplicates(ev, threshold=0.8):
    print(pair)   # reset-pw ~ reset-pw-dupe (similarity=1.000)
```

The check is O(n^2) over case pairs — fine for eval-sized sets (thousands of
cases). Note that `Eval` construction already rejects *identical* duplicate
cases at the door; this check catches the near-misses that survive.

## The HoldoutLedger

Every time a team tunes a prompt against the same eval set, the eval stops
measuring generalization and starts measuring memorization-by-iteration — the
silent killer quantitative finance calls backtest overfitting. The effect is
formal, not folklore: each adaptive look at the same data biases the next
decision, and the bias compounds with the number of looks (Dwork et al. 2015,
"The reusable holdout", *Science* 349(6248); Russo & Zou 2016, "Controlling
bias in adaptive data analysis", AISTATS).

The ledger cannot stop you from peeking. It makes the peeking visible:

```python
from holdout.leakage import HoldoutLedger

ledger = HoldoutLedger(".holdout")            # lives next to the run store
ledger.record_use(ev.fingerprint, ev.name, kind="compare", context="PR #412")
report = ledger.check(ev.fingerprint, ev.name, budget=20)
print(report)
# eval 'support-qa' has been used 14 time(s) of a budget of 20 [caution] — plan
# a fresh holdout set before the budget runs out
```

Uses are counted per eval *fingerprint* — the content hash of the dataset — so
editing the cases starts a fresh count, and renaming the eval does not. The
discipline levels:

| Level | Condition | Reading |
|---|---|---|
| `ok` | under half the budget | verdicts still mean what they say |
| `caution` | half the budget or more | plan a fresh holdout set |
| `overfit-risk` | budget spent | results now reflect tuning-to-the-test as much as quality; cut a fresh set |

The default budget of 20 is generous — quant desks would say lower. Every
`holdout compare` records one use automatically (opt out with `--no-ledger`)
and prints the ledger line under the verdict, so the wear on your eval set is
in your face in CI, not in a dashboard nobody opens.

## assert_no_leakage

The pytest-facing wrapper bundles the offline checks:

```python
from holdout.testing import assert_no_leakage

def test_eval_hygiene():
    assert_no_leakage(ev, SYSTEM_PROMPT)
```

It checks every case against the corpus (exact-substring + n-gram containment)
and, by default, also fails on near-duplicate pairs inside the eval
(`duplicate_threshold=0.8`; pass `None` to skip). The corpus can be a string, a
list of strings, or a `Target` — a provider's `system` prompt is extracted
automatically; targets without one raise `ValueError` and you pass the text
explicitly. On failure the `AssertionError` carries the full report:

```text
AssertionError: eval leakage detected:
contamination check (exact-substring + 5-gram containment >= 0.5): 2/4 case(s) flagged
  reset-pw.input: exact-substring (score=1.000) — appears verbatim in corpus text #0
  ...
near-duplicate cases inflate the effective sample size (n=4 is overstated):
  reset-pw ~ reset-pw-dupe (similarity=1.000)
```

## holdout check from the CLI

The same audit, CI-ready (exit 0 = clean, 1 = findings):

```console
$ holdout check qa.jsonl --corpus system_prompt.txt
contamination check (exact-substring + 5-gram containment >= 0.5): 2/4 case(s) flagged
  reset-pw.input: exact-substring (score=1.000) — appears verbatim in corpus text #0
  reset-pw.reference: exact-substring (score=1.000) — appears verbatim in corpus text #0
near-duplicate cases (effective n is below 4):
  reset-pw ~ reset-pw-dupe (similarity=1.000)
$ echo $?
1
```

`--corpus` takes a text file and is repeatable; `--corpus-text` passes literal
prompt text; `--ngram`, `--threshold`, and `--dup-threshold` tune the knobs.
With no corpus given, the contamination pass is skipped (and says so) and only
the duplicate check runs. Details in the [CLI reference](cli.md#holdout-check).
