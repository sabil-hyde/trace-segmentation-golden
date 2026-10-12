"""MinHasher from trace_mining/common.py (unchanged): word-shingle MinHash with LSH banding."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Set, Tuple


class MinHasher:
    """Word-shingle MinHash with LSH banding to find near-duplicate texts."""

    def __init__(self, num_perm: int = 64, shingle: int = 5, bands: int = 8):
        assert num_perm % bands == 0
        self.num_perm = num_perm
        self.shingle = shingle
        self.bands = bands
        self.rows = num_perm // bands

    def signature(self, text: str) -> Optional[Tuple[int, ...]]:
        words = text.lower().split()
        if len(words) < self.shingle:
            shingles = {" ".join(words)} if words else set()
        else:
            shingles = {
                " ".join(words[i : i + self.shingle])
                for i in range(len(words) - self.shingle + 1)
            }
        if not shingles:
            return None
        sig = []
        for p in range(self.num_perm):
            best = min(
                int.from_bytes(
                    hashlib.blake2b(
                        s.encode(), digest_size=8, salt=p.to_bytes(2, "big")
                    ).digest(),
                    "big",
                )
                for s in shingles
            )
            sig.append(best)
        return tuple(sig)

    @staticmethod
    def jaccard(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
        return sum(1 for x, y in zip(a, b) if x == y) / len(a)

    def dedup_ids(
        self, items: List[Tuple[str, str]], threshold: float
    ) -> Set[str]:
        """Return the set of ids to DROP (near-dups of an earlier item).

        items: list of (id, text), order defines which copy survives.
        """
        buckets: Dict[Tuple[int, ...], List[int]] = {}
        sigs: List[Optional[Tuple[int, ...]]] = []
        drop: Set[str] = set()
        for idx, (_id, text) in enumerate(items):
            sig = self.signature(text)
            sigs.append(sig)
            if sig is None:
                continue
            is_dup = False
            candidate_idxs: Set[int] = set()
            for b in range(self.bands):
                band = (b,) + sig[b * self.rows : (b + 1) * self.rows]
                for j in buckets.get(band, []):
                    candidate_idxs.add(j)
            for j in candidate_idxs:
                other = sigs[j]
                if other and self.jaccard(sig, other) >= threshold:
                    is_dup = True
                    break
            if is_dup:
                drop.add(_id)
                continue
            for b in range(self.bands):
                band = (b,) + sig[b * self.rows : (b + 1) * self.rows]
                buckets.setdefault(band, []).append(idx)
        return drop
