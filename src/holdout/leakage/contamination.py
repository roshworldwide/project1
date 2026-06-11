"""Contamination detection — finding eval data hiding inside the prompt.

The quiet way an eval lies: a case (or its answer) appears verbatim or
near-verbatim in the system prompt or few-shot examples, and the model
"solves" it by recall. These checks compare every case's input and
reference against a corpus of prompt/few-shot text, with exact-substring
matching, word n-gram containment (Brown et al. 2020, app. C), and an
optional embedding pass for paraphrase-level leakage.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from holdout.core.evalset import Eval
from holdout.leakage.ngram import containment, tokens, word_ngrams
from holdout.scorers.embedding import EmbeddingBackend, cosine_similarity

ContaminationKind = Literal["exact-substring", "ngram-overlap", "embedding-similarity"]


@dataclass(frozen=True, slots=True)
class ContaminationFinding:
    """One contaminated case field.

    Parameters
    ----------
    case_id
        The affected case.
    field
        Which side leaked: ``"input"`` or ``"reference"``.
    kind
        How it was caught.
    score
        Containment fraction or cosine similarity (1.0 for exact matches).
    detail
        Human-readable context.
    """

    case_id: str
    field: Literal["input", "reference"]
    kind: ContaminationKind
    score: float
    detail: str

    def __str__(self) -> str:
        return f"{self.case_id}.{self.field}: {self.kind} (score={self.score:.3f}) — {self.detail}"


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    """The outcome of a contamination check.

    Parameters
    ----------
    findings
        Contaminated case fields, worst first.
    n_cases
        Number of cases checked.
    method
        Description of the check and its parameters.
    """

    findings: tuple[ContaminationFinding, ...]
    n_cases: int
    method: str

    @property
    def clean(self) -> bool:
        """True when nothing leaked."""
        return not self.findings

    @property
    def contaminated_case_ids(self) -> tuple[str, ...]:
        """Distinct ids of affected cases, in finding order."""
        seen: dict[str, None] = {}
        for f in self.findings:
            seen.setdefault(f.case_id, None)
        return tuple(seen)

    def summary(self) -> str:
        """Render a human-readable report."""
        head = (
            f"contamination check ({self.method}): "
            f"{len(self.contaminated_case_ids)}/{self.n_cases} case(s) flagged"
        )
        lines = [head]
        lines.extend(f"  {f}" for f in self.findings)
        if self.clean:
            lines.append("  no contamination detected")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "n_cases": self.n_cases,
            "method": self.method,
            "clean": self.clean,
            "findings": [
                {
                    "case_id": f.case_id,
                    "field": f.field,
                    "kind": f.kind,
                    "score": f.score,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }


def _as_corpus(corpus: str | Sequence[str]) -> list[str]:
    return [corpus] if isinstance(corpus, str) else list(corpus)


def check_contamination(
    ev: Eval,
    corpus: str | Sequence[str],
    *,
    ngram_size: int = 5,
    threshold: float = 0.5,
    min_tokens: int = 3,
) -> ContaminationReport:
    """Check every case against prompt/few-shot text, no model required.

    Two passes per case field (input and reference):

    1. **Exact substring** — the field's word sequence appears verbatim in
       a corpus text (score 1.0). Matching is at word boundaries on
       casefolded, punctuation-stripped tokens, so a one-letter reference
       cannot "match" inside an unrelated word.
    2. **N-gram containment** — at least ``threshold`` of the field's word
       ``ngram_size``-grams appear in a corpus text (Brown et al. 2020,
       app. C, adapted for prompt-sized corpora).

    Fields with fewer than ``min_tokens`` words are skipped for the n-gram
    pass (a two-word answer matching a prompt is noise, not leakage) but
    still checked for exact substring presence.

    Parameters
    ----------
    ev
        The eval to audit.
    corpus
        Prompt text(s) to check against — a system prompt, few-shot block,
        or any strings that will be shown to the model.
    ngram_size
        Word n-gram size (default 5).
    threshold
        Containment fraction at or above which a field is flagged.
    min_tokens
        Minimum field length (in tokens) for the n-gram pass.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    texts = _as_corpus(corpus)
    # Word-boundary canon: " a b c " so substring hits are whole-token runs.
    padded = [" " + " ".join(tokens(t)) + " " for t in texts]
    corpus_grams = [word_ngrams(t, ngram_size) for t in texts]

    findings: list[ContaminationFinding] = []
    for case in ev.cases:
        assert case.id is not None  # Eval normalization guarantees ids
        fields: list[tuple[Literal["input", "reference"], str]] = [("input", case.input)]
        if case.reference is not None:
            fields.append(("reference", case.reference))
        for field_name, text in fields:
            toks = tokens(text)
            if not toks:
                continue
            needle = " " + " ".join(toks) + " "
            hit = next((i for i, t in enumerate(padded) if needle in t), None)
            if hit is not None:
                findings.append(
                    ContaminationFinding(
                        case_id=case.id,
                        field=field_name,
                        kind="exact-substring",
                        score=1.0,
                        detail=f"appears verbatim in corpus text #{hit}",
                    )
                )
                continue
            grams = word_ngrams(text, ngram_size)
            if len(toks) < min_tokens or not grams:
                continue
            best, best_idx = 0.0, -1
            for i, cg in enumerate(corpus_grams):
                c = containment(grams, cg)
                if c > best:
                    best, best_idx = c, i
            if best >= threshold:
                findings.append(
                    ContaminationFinding(
                        case_id=case.id,
                        field=field_name,
                        kind="ngram-overlap",
                        score=best,
                        detail=(
                            f"{best:.0%} of its {ngram_size}-grams appear in corpus "
                            f"text #{best_idx}"
                        ),
                    )
                )
    findings.sort(key=lambda f: -f.score)
    return ContaminationReport(
        findings=tuple(findings),
        n_cases=len(ev.cases),
        method=f"exact-substring + {ngram_size}-gram containment >= {threshold:g}",
    )


async def check_contamination_embeddings(
    ev: Eval,
    corpus: str | Sequence[str],
    backend: EmbeddingBackend,
    *,
    threshold: float = 0.9,
) -> ContaminationReport:
    """Embedding pass: catches paraphrase-level leakage n-grams miss.

    Embeds every case input/reference and every corpus text, flagging
    fields whose cosine similarity to any corpus text meets ``threshold``.
    Use a local backend (Ollama) to keep the audit air-gapped.

    Parameters
    ----------
    ev
        The eval to audit.
    corpus
        Prompt text(s) to check against.
    backend
        The embedding backend.
    threshold
        Cosine similarity at or above which a field is flagged.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    texts = _as_corpus(corpus)
    if not texts:
        raise ValueError("corpus is empty")

    fields: list[tuple[str, Literal["input", "reference"], str]] = []
    for case in ev.cases:
        assert case.id is not None
        fields.append((case.id, "input", case.input))
        if case.reference is not None:
            fields.append((case.id, "reference", case.reference))

    vectors = await backend.embed([text for _, _, text in fields] + texts)
    field_vecs = vectors[: len(fields)]
    corpus_vecs = vectors[len(fields) :]

    findings: list[ContaminationFinding] = []
    for (case_id, field_name, _), vec in zip(fields, field_vecs, strict=True):
        best, best_idx = -1.0, -1
        for i, cv in enumerate(corpus_vecs):
            sim = cosine_similarity(vec, cv)
            if sim > best:
                best, best_idx = sim, i
        if best >= threshold:
            findings.append(
                ContaminationFinding(
                    case_id=case_id,
                    field=field_name,
                    kind="embedding-similarity",
                    score=best,
                    detail=f"cosine {best:.3f} to corpus text #{best_idx} ({backend.name})",
                )
            )
    findings.sort(key=lambda f: -f.score)
    return ContaminationReport(
        findings=tuple(findings),
        n_cases=len(ev.cases),
        method=f"embedding cosine >= {threshold:g} ({backend.name})",
    )
