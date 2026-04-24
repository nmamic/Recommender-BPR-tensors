#TD / Tuckerova dekompozicija

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from TagRecommender.bazniModel import BazniOznakaModel, bpr_delta
from TagRecommender.data import TenzorPodaci
from TagRecommender.uzorkovanje import BPRUzorkovanje

class TuckerDekompozicija(BazniOznakaModel):
    """
    Tuckerova dekompozicija :
        y(u,i,t) = sum_{a,b,c} C[a,b,c] * U[u,a] * I[i,b] * T[t,c]
    3D jezgreni tenzor...

    Veličine:
        U: (n_korisnici, k_u)
        I: (n_artikli, k_i)
        T: (n_oznake,  k_t)
        C: (k_u, k_i, k_t)
    """
    def __init__(self,
                 n_korisnici: int, n_artikli: int, n_oznake: int,
                 k_u: int, k_i: int, k_t: int,
                 init_std: float = 0.01, seed: int = 42,
                 optimizator: str | None = None,
                 gama: float = 1e-6,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8):

        if optimizator not in (None, "decay", "adam", "adagrad"):
            raise ValueError("optimizator mora biti None, 'decay', 'adam' ili 'adagrad'")

        rng = np.random.default_rng(seed)
        self.k_u, self.k_i, self.k_t = int(k_u), int(k_i), int(k_t)

        self.U = rng.normal(0.0, init_std, size=(n_korisnici, k_u)).astype(np.float64)
        self.I = rng.normal(0.0, init_std, size=(n_artikli, k_i)).astype(np.float64)
        self.T = rng.normal(0.0, init_std, size=(n_oznake,  k_t)).astype(np.float64)

        # 3D jezgreni tenzor
        self.C = rng.normal(0.0, init_std, size=(k_u, k_i, k_t)).astype(np.float64)

        #parametri za optimizatore
        self.optimizator = optimizator
        self.gamma = float(gama)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        # Adam stanje
        self._adam_inicijaliziran = False
        self.t_adam = 0

        # Adagrad stanje
        self._adagrad_initialized = False

    def score(self, u: int, i: int, t: int) -> float:
        #einsum (Einsteinova sumacija) je kompaktan način zapisa matrično-tenzorskih operacija
        #npr ovdje "a,b,c,abc->" znači "hodam po prvom parametru indeksom a, po drugom indeksom b, po trećem indeksom c,
        #a po četvrtom (3D tenzor) indeksima a,b,c. Produkt sva četiri elementa i sumiraj po indeksima s lijeve strane"
        #dakle kompaktan način zapisati \sum_{a,b,c} (U[u,a] * I[i,b] * T[t,c] * C[a,b,c]) (jednakost (9) iz papera)
        return float(np.einsum("a,b,c,abc->", self.U[u], self.I[i], self.T[t], self.C))

    def ocijeni_sve_oznake(self, u: int, i: int) -> np.ndarray:
        #ovdje pak c s desne strane strelice znači "ne sumiraj po c, nego napravi ovo slijeva za fiksne c-ove"
        #u stvari mu kažeš "vraćaš 1D vektor po indeksu c"
        #kompaktan način zapisati g[c] = \sum_{a,b} (U[u,a] * I[i,b] * C[a,b,c])
        #u stvari tu radim modmult(2, modmult(1, C, U[u]), I[i] )
        g = np.einsum("a,b,abc->c", self.U[u], self.I[i], self.C)  # (k_t,)
        #ovo je rezultati[t] = \sum_{c} (T[t,c] * g[c])
        #tu je onaj zadnji modmult sa T iz jednakosti (10) iz papera
        return self.T @ g

    def _inicijaliz_adam_stanje(self):
        self.mU = np.zeros_like(self.U); self.vU = np.zeros_like(self.U)
        self.mI = np.zeros_like(self.I); self.vI = np.zeros_like(self.I)
        self.mT = np.zeros_like(self.T); self.vT = np.zeros_like(self.T)
        self.mC = np.zeros_like(self.C); self.vC = np.zeros_like(self.C)
        self._adam_inicijaliziran = True
        self.t_adam = 0

    def _inicijaliz_adagrad_state(self):
        self.gU = np.zeros_like(self.U)
        self.gI = np.zeros_like(self.I)
        self.gT = np.zeros_like(self.T)
        self.gC = np.zeros_like(self.C)
        self._adagrad_initialized = True

    def fit_bpr(self, data: TenzorPodaci, steps: int = 100_000,
                lr: float = 0.02, reg: float = 1e-5,
                seed: int = 42, progress: bool = True) -> "TuckerDekompozicija":

        sampler = BPRUzorkovanje(data, seed=seed)
        it = range(steps)
        if progress:
            it = tqdm(it, desc="BPR-TD", unit="step")

        for step in it:
            u, i, t_pos, t_neg = sampler.uzorkuj()

            #x = \hat{y}_{u,i,t+} - \hat{y}_{u,i,t-}
            x = self.score(u, i, t_pos) - self.score(u, i, t_neg)
            #d = 1 - sigmoida(x)
            d = bpr_delta(x)

            if self.optimizator == "decay":
                lr_t = lr / (1.0 + self.gamma * step)
            else:
                lr_t = lr

            #cachiram potrebne vektore
            Uu = self.U[u].copy()          # (k_u,)
            Ii = self.I[i].copy()          # (k_i,)
            Tpos = self.T[t_pos].copy()    # (k_t,)
            Tneg = self.T[t_neg].copy()    # (k_t,)
            C = self.C                     # (k_u,k_i,k_t)

            #izračunam ponovljene međurezultate
            # g_pos[c] = \sum_{a,b} Uu[a] * Ii[b] * C[a,b,c]  ( ovo je parc \hat{y} / T[t,c])
            g_ui = np.einsum("a,b,abc->c", Uu, Ii, C)  # (k_t,)

            # parc \hat{y} / U[u,a] = \sum_{b,c} C[a,b,c] * Ii[b] * T[t,c]
            dU_pos = np.einsum("b,c,abc->a", Ii, Tpos, C)  # (k_u,)
            dU_neg = np.einsum("b,c,abc->a", Ii, Tneg, C)  # (k_u,)

            #parc \hat{y}/I[i,b] = \sum_{a,c} C[a,b,c] * Uu[a] * T[t,c]
            dI_pos = np.einsum("a,c,abc->b", Uu, Tpos, C)  # (k_i,)
            dI_neg = np.einsum("a,c,abc->b", Uu, Tneg, C)  # (k_i,)

            # parc \hat{y}/T[t,c] = g_ui[c]
            dT_zajednicki = g_ui  # (k_t,)

            # parc \hat{y}/C[a,b,c] = Uu[a] * Ii[b] * T[t,c]
            dC_pos = np.einsum("a,b,c->abc", Uu, Ii, Tpos)
            dC_neg = np.einsum("a,b,c->abc", Uu, Ii, Tneg)

            
            # sad sastavljam BPR gradijent
            #L(theta) = log(sigmoida(x)) - reg * ||theta||^2
            grad_U = d * (dU_pos - dU_neg) - reg * Uu
            grad_I = d * (dI_pos - dI_neg) - reg * Ii

            grad_T_pos = d * dT_zajednicki - reg * Tpos
            grad_T_neg = -d * dT_zajednicki - reg * Tneg

            grad_C = d * (dC_pos - dC_neg) - reg * self.C

            if self.optimizator == "adam":
                if not self._adam_inicijaliziran:
                    self._inicijaliz_adam_stanje()

                self.t_adam += 1
                t = self.t_adam
                b1, b2 = self.beta1, self.beta2

                # U[u]
                self.mU[u] = b1*self.mU[u] + (1-b1)*grad_U
                self.vU[u] = b2*self.vU[u] + (1-b2)*(grad_U**2)
                m_hat = self.mU[u] / (1 - b1**t)
                v_hat = self.vU[u] / (1 - b2**t)
                self.U[u] += lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

                # I[i]
                self.mI[i] = b1*self.mI[i] + (1-b1)*grad_I
                self.vI[i] = b2*self.vI[i] + (1-b2)*(grad_I**2)
                m_hat = self.mI[i] / (1 - b1**t)
                v_hat = self.vI[i] / (1 - b2**t)
                self.I[i] += lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

                # T[t_pos]
                self.mT[t_pos] = b1*self.mT[t_pos] + (1-b1)*grad_T_pos
                self.vT[t_pos] = b2*self.vT[t_pos] + (1-b2)*(grad_T_pos**2)
                m_hat = self.mT[t_pos] / (1 - b1**t)
                v_hat = self.vT[t_pos] / (1 - b2**t)
                self.T[t_pos] += lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

                # T[t_neg]
                self.mT[t_neg] = b1*self.mT[t_neg] + (1-b1)*grad_T_neg
                self.vT[t_neg] = b2*self.vT[t_neg] + (1-b2)*(grad_T_neg**2)
                m_hat = self.mT[t_neg] / (1 - b1**t)
                v_hat = self.vT[t_neg] / (1 - b2**t)
                self.T[t_neg] += lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

                # C (PUNI TENZOR)
                self.mC = b1*self.mC + (1-b1)*grad_C
                self.vC = b2*self.vC + (1-b2)*(grad_C**2)
                m_hat = self.mC / (1 - b1**t)
                v_hat = self.vC / (1 - b2**t)
                self.C += lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

            elif self.optimizator == "adagrad":
                if not self._adagrad_initialized:
                    self._inicijaliz_adagrad_state()

                self.gU[u] += grad_U**2
                self.gI[i] += grad_I**2
                self.gT[t_pos] += grad_T_pos**2
                self.gT[t_neg] += grad_T_neg**2
                self.gC += grad_C**2

                self.U[u] += lr_t * grad_U / (np.sqrt(self.gU[u]) + self.eps)
                self.I[i] += lr_t * grad_I / (np.sqrt(self.gI[i]) + self.eps)
                self.T[t_pos] += lr_t * grad_T_pos / (np.sqrt(self.gT[t_pos]) + self.eps)
                self.T[t_neg] += lr_t * grad_T_neg / (np.sqrt(self.gT[t_neg]) + self.eps)
                self.C += lr_t * grad_C / (np.sqrt(self.gC) + self.eps)

            else:
                #običan SGD
                self.U[u] += lr_t * grad_U
                self.I[i] += lr_t * grad_I
                self.T[t_pos] += lr_t * grad_T_pos
                self.T[t_neg] += lr_t * grad_T_neg
                self.C += lr_t * grad_C

        return self
