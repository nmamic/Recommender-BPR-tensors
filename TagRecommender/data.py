#osnovni datatype i pretprocesiranje, p-core filtriranje podataka

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

#ovo je dekorator iz standardne python biblioteke, automatski napravi __init__ i ostali boilerplate kod
@dataclass
class TenzorPodaci:
    """
    Sprema tenzor s oznakama kao promatrane trojke S podskup UxIxT

    trojke: lista (u, i, t)-ova s cjelobrojnim indeksima iz [0..n_k), [0..n_a), [0..n_t)
    """
    n_korisnici: int
    n_artikli: int
    n_oznake: int
    trojke: np.ndarray  # oblika (n_obs, 3), dtype int64

    def sagradi_post_index(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Mapiraj svaki post (u,i) polju promatranih oznaka za taj post.
        """
        post_oznake: Dict[Tuple[int, int], List[int]] = {}
        for u, i, t in self.trojke:
            key = (int(u), int(i))
            post_oznake.setdefault(key, []).append(int(t))
        # unique tags per post for correct negative sampling
        return {k: np.array(sorted(set(v)), dtype=np.int64) for k, v in post_oznake.items()}


def p_core_filter(trojke, p=5):
    promjena_se_dogodila = True

    # jer kad nesto maknem, promijene se potencijalno i ostali zapisi...
    while promjena_se_dogodila:
        promjena_se_dogodila = False

        # broj postova po korisniku (distinct (u,i))
        korisnici_postovi = {}
        for u, i, t in trojke:
            korisnici_postovi.setdefault(u, set()).add(i)
        korisnik_count = {u: len(postovi) for u, postovi in korisnici_postovi.items()}

        # broj postova po artiklu (distinct (u,i))
        artikl_postovi = {}
        for u, i, t in trojke:
            artikl_postovi.setdefault(i, set()).add(u)
        artikl_count = {i: len(postovi) for i, postovi in artikl_postovi.items()}

        # broj postova po oznaci (distinct (u,i))
        oznaka_postovi = {}
        for u, i, t in trojke:
            oznaka_postovi.setdefault(t, set()).add((u, i))
        oznaka_count = {t: len(postovi) for t, postovi in oznaka_postovi.items()}

        # maska
        mask = []
        for u, i, t in trojke:
            if (
                korisnik_count.get(u, 0) >= p and
            artikl_count.get(i, 0) >= p and
                oznaka_count.get(t, 0) >= p
            ):
                mask.append(True)
            else:
                mask.append(False)

        mask = np.array(mask)

        if not np.all(mask):
            trojke = trojke[mask]
            promjena_se_dogodila = True

    return trojke


# Ovdje radim podjelu i topN F1 kao u originalnom radu:

def train_test_podjela_po_postovima_korisnika(data: TenzorPodaci, seed: int = 42) -> Tuple[TenzorPodaci, TenzorPodaci]:
    """
    Split kao u paperu - za svakog korisnika uklanjam 1 post i stavljam u testni skup.
    """
    rng = np.random.default_rng(seed)
    trojke = data.trojke
    #gradim postove po svakom korisniku
    postovi_po_korisniku: Dict[int, List[Tuple[int,int]]] = {}
    for u, i, t in trojke:
        #za svakog korisnika svi njegovi postovi
        #znaci poslije ovog imas rjecnik korisnik_i -> lista postova oblika (korisnik_i, proizvod/item)
        #stavljam u skup t.d. ako ima jedan post više tagova (znači isti post se pojavljuje više puta kao trojka),
        #npr (u1,i1,t1), (u1,i1,t2) -> oba se stave u postovi_po_korisniku ali nema duplikacije jer je Set
        #pa ostane samo (u1,i1)
        postovi_po_korisniku.setdefault(int(u), set()).add((int(u), int(i)))
    #sad kad sam sve prošao prebacim skupove postova u listu    
    postovi_po_korisniku = {u: list(postovi) for u, postovi in postovi_po_korisniku.items()}

    #nasumnicno biram jedan post i stavljam u skup za testiranje
    test_postovi = set()
    for u, postovi in postovi_po_korisniku.items():
        if len(postovi) > 0:
            test_postovi.add(postovi[rng.integers(0, len(postovi))])

    #sad napravim masku - ide po svim trojkama - ako je post u testnom skupu, oznaci taj indeks sa True, inace s False 
    #znaci mask_test je array velicine array-a trojke, koji govori koje trojke idu u testni, koji u train skup
    mask_test = np.array([(int(u), int(i)) in test_postovi for u, i, _ in trojke], dtype=bool)
    #trojke za treniranje su one koje nisu upale u masku
    train_trojke = trojke[~mask_test]
    test_trojke = trojke[mask_test]

    #i strpam to sve u objekt da lakse radim
    train = TenzorPodaci(data.n_korisnici, data.n_artikli, data.n_oznake, train_trojke)
    test  = TenzorPodaci(data.n_korisnici, data.n_artikli, data.n_oznake, test_trojke)
    return train, test

#remapira originalne IDjeve na 0-indeksirane indekse
#prima trojke - oblika (N,3)
def remapiraj_indekse(trojke: np.ndarray) -> Tuple[np.ndarray, dict, dict, dict]:
    korisnici = {u: i for i, u in enumerate(np.unique(trojke[:, 0]))}
    artikli = {v: i for i, v in enumerate(np.unique(trojke[:, 1]))}
    oznake  = {t: i for i, t in enumerate(np.unique(trojke[:, 2]))}

    mapped = np.zeros_like(trojke)
    for idx, (u, i, t) in enumerate(trojke):
        mapped[idx, 0] = korisnici[u]
        mapped[idx, 1] = artikli[i]
        mapped[idx, 2] = oznake[t]

    return mapped, korisnici, artikli, oznake