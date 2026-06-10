"""Canonical serialization and content fingerprints.

Everything in holdout that claims an identity — evals, targets, scorers,
runs — derives it from a SHA-256 hash of canonical JSON. Identical content
always produces identical identifiers; this is the foundation of the
determinism guarantee (same seed + same inputs => identical run hash).
"""

import hashlib
import json


def canonical_json(obj: object) -> str:
    """Serialize ``obj`` to canonical JSON: sorted keys, no whitespace.

    Parameters
    ----------
    obj
        Any JSON-serializable object.

    Returns
    -------
    str
        A canonical, byte-stable JSON string.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(obj: object) -> str:
    """Return the SHA-256 hex digest of the canonical JSON of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def short_id(full: str, length: int = 12) -> str:
    """Return a short display prefix of a full hex fingerprint."""
    return full[:length]
