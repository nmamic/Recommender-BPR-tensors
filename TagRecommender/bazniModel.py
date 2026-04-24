# bazna klasa za modele, osnovne funkcije za korištenje u svim modelima

from __future__ import annotations

import math

import numpy as np

def sigmoid(x: float) -> float:
    # numericki stabilna sigmoidna fja
    #za jako male/jako velike x-eve da sprijecim over/underflow
    if x >= 0:
        # za x>=0 
        #sigmoid(x) = 1/(1+e^-x)
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        #a za vrlo velike negativne x bi moglo biti e^-x ogromno
        #pa zato radim sigmoid(x) = e^x / (1 + e^x)
        z = math.exp(x)
        return z / (1.0 + z)

def bpr_delta(x_uij: float) -> float:
    # delta = (1 - sigmoida(x_uij)) gdje x_uij = rezultat_poz - rezultat_neg
    return 1.0 - float(sigmoid(x_uij))


#bazna klasa, sve izvedene implementiraju ove metode
class BazniOznakaModel:
    def score(self, u: int, i: int, t: int) -> float:
        raise NotImplementedError

    def ocijeni_sve_oznake(self, u: int, i: int) -> np.ndarray:
        """
        Vektor ocjena za sve oznake za fiksan post (u,i).
        """
        raise NotImplementedError

    def predlozi(self, u: int, i: int, topn: int = 10) -> np.ndarray:
        #predlaze topn oznaka za post (u,i)
        scores = self.ocijeni_sve_oznake(u, i)
        if topn >= len(scores):
            return np.argsort(-scores)
        # partial top-k
        idx = np.argpartition(-scores, topn)[:topn]
        idx = idx[np.argsort(-scores[idx])]
        return idx