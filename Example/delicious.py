# podaci iz delicious folksonomy skupa podataka, usporedba PARAFAC i PITF modela na 3-core filtriranim podacima

import numpy as np
import pandas as pd

from TagRecommender import (
    TenzorPodaci,
    p_core_filter,
    remapiraj_indekse,
    train_test_podjela_po_postovima_korisnika,
    metrike_at_n,
    bazna_preciznost,
    PITF,
    KanonskaDekompozicija,
)

DATA_PATH = "Data/user_taggedbookmarks.dat"
P_CORE    = 3
TOPN      = 3


def load_delicious(path: str, p: int = 3):
    df = pd.read_csv(path, sep="\t", header=0)
    trojke = df[["userID", "bookmarkID", "tagID"]].values.astype(np.int64)
    print(f"Trojke nefiltrirane: {len(trojke):,}")

    trojke = p_core_filter(trojke, p=p)
    print(f"Nakon {p}-core filtriranja: {len(trojke):,}")

    mapped, korisnici, artikli, oznake = remapiraj_indekse(trojke)
    print(f"Korisnici: {len(korisnici)}, artikli: {len(artikli)}, oznake: {len(oznake)}")

    return TenzorPodaci(
        n_korisnici=len(korisnici),
        n_artikli=len(artikli),
        n_oznake=len(oznake),
        trojke=mapped,
    )

def main():
    data = load_delicious(DATA_PATH, p=P_CORE)
    train, test = train_test_podjela_po_postovima_korisnika(data)

    print("\n--- Nasumično baseline biranje ---")
    rand_p = bazna_preciznost(test, n=TOPN)
    print(f"Random Prec@{TOPN} = {rand_p:.4f}")

    print("\n--- PITF (k=128, AdaGrad, 500k koraka) ---")
    pitf = PITF(
        train.n_korisnici, train.n_artikli, train.n_oznake,
        k=128, optimizator="adagrad",
    )
    pitf.fit_bpr(train, steps=500_000, lr=0.05, reg=1e-5)
    p, r, f1 = metrike_at_n(pitf, test, n=TOPN)
    print(f"Prec@{TOPN}={p:.4f}  Recall@{TOPN}={r:.4f}  F1@{TOPN}={f1:.4f}")

    print("\n--- CP/PARAFAC (k=128, AdaGrad, 700k koraka) ---")
    cd = KanonskaDekompozicija(
        train.n_korisnici, train.n_artikli, train.n_oznake,
        k=128, optimizator="adagrad",
    )
    cd.fit_bpr(train, steps=700_000, lr=0.01, reg=0)
    p, r, f1 = metrike_at_n(cd, test, n=TOPN)
    print(f"Prec@{TOPN}={p:.4f}  Recall@{TOPN}={r:.4f}  F1@{TOPN}={f1:.4f}")


if __name__ == "__main__":
    main()