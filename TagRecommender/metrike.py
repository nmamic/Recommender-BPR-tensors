#metrike za evaluaciju rezultata  predlaganja

from __future__ import annotations

from typing import Tuple

import numpy as np

from TagRecommender.bazniModel import BazniOznakaModel
from TagRecommender.data import TenzorPodaci


def metrike_at_n(model: BazniOznakaModel, test: TenzorPodaci, n: int = 5) -> Tuple[float, float, float]:
    #racunam f1@n
    #mapiram sve postove njihovim oznakama
    test_post_oznake = test.sagradi_post_index()
    preciznosti = []
    recallovi = []
    for (u, i), true_oznake in test_post_oznake.items():
        #gledam koliko tocnih je predlozio u top n predlozenih
        rec = model.predlozi(u, i, topn=n)
        true_set = set(map(int, true_oznake))
        hit = sum((int(t) in true_set) for t in rec)
        preciznosti.append(hit / float(n))
        recallovi.append(hit / float(len(true_set)) if len(true_set) else 0.0)

    #prosjecna preciznost, prosjecni recall
    p = float(np.mean(preciznosti)) if preciznosti else 0.0
    r = float(np.mean(recallovi)) if recallovi else 0.0
    #f1@n u prosjeku
    return p, r, (2*p*r/(p+r)) if (p+r) > 0 else 0.0



#preciznost za nasumično predlaganje, za sanity-check
def bazna_preciznost(test: TenzorPodaci, n: int = 5, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    postovi = list(test.sagradi_post_index().items())

    hits = total = 0
    #uzimam nasumicne za sve oznake, usporedujem sa stvarnim...
    for (u, i), true_tags in postovi:
        true_set = set(map(int, true_tags))
        rec = rng.choice(test.n_oznake, size=n, replace=False)
        hits  += sum(int(t) in true_set for t in rec)
        total += n

    return hits / total if total > 0 else 0.0