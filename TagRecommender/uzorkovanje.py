from __future__ import annotations

from typing import Tuple

import numpy as np

from TagRecommender.data import TenzorPodaci

class BPRUzorkovanje:
    """
    Uzorkuje četvorke (u, i, t_pos, t_neg) uniformno iz D_S.
    Negativne oznake se uzorkuju iz svih oznaka koje nisu u promatranom skupu oznaka za post (skup S).
    """
    def __init__(self, data: TenzorPodaci, seed: int = 42):
        self.data = data
        self.rng = np.random.default_rng(seed)
        self.post_oznake = data.sagradi_post_index() # rječnik oblika (u,i) -> [t1, t2, ...]
        self.postovi = np.array(list(self.post_oznake.keys()), dtype=np.int64)  # oblika (n_postova, 2) - svi postovi, npr [(0,2), (0,4), (3,1), ...]
        self.sve_oznake = np.arange(data.n_oznake, dtype=np.int64)
        self._oznake_skupovi = {k: set(map(int, v)) for k, v in self.post_oznake.items()} #za O(1) lookup "je li oznaka i pozotivna za post j?"

    def uzorkuj(self) -> Tuple[int, int, int, int]:
        u, i = self.postovi[self.rng.integers(0, len(self.postovi))] #uniformno odabire post
        oznake_pos = self.post_oznake[(int(u), int(i))] #dohvatim pozitivne oznake za post
        t_pos = int(oznake_pos[self.rng.integers(0, len(oznake_pos))]) #odabere nasumičnu pozitivnu oznaku za post (postoji jer je p-core skup podataka)

        # sad biram negativnu oznaku (u,i,t_neg) nije element od S
        tagset = self._oznake_skupovi[(int(u), int(i))]
        while True:
            t_neg = int(self.rng.integers(0, self.data.n_oznake)) #nasumicni indeks oznake
            if t_neg not in tagset: 
                break
        return int(u), int(i), t_pos, t_neg
