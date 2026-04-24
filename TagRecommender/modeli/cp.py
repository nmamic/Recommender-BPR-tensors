#CP / kanonska dekompozicija

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from TagRecommender.bazniModel import BazniOznakaModel, bpr_delta
from TagRecommender.data import TenzorPodaci
from TagRecommender.uzorkovanje import BPRUzorkovanje


class KanonskaDekompozicija(BazniOznakaModel):
    """
    PARAFAC:
        \hat{y}(u,i,t) = sum(od f do k) U[u,f] * I[i,f] * T[t,f]
        (k je broj latentnih faktora)
    """
    def __init__(self,
                 n_korisnici: int,
                 n_artikli: int,
                 n_oznake: int,
                 k: int,
                 init_std: float = 0.01,
                 seed: int = 42,
                 optimizator: str | None = None,
                 gama: float = 1e-6,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8):

        if optimizator not in (None, "decay", "adam", "adagrad"):
            raise ValueError("optimizator mora biti None, 'decay', 'adam' ili 'adagrad'")

        rng = np.random.default_rng(seed)
        self.k = int(k)

        self.U = rng.normal(0.0, init_std, size=(n_korisnici, k)).astype(np.float64)
        self.I = rng.normal(0.0, init_std, size=(n_artikli, k)).astype(np.float64)
        self.T = rng.normal(0.0, init_std, size=(n_oznake,  k)).astype(np.float64)

        # hiperparametri optimizatora
        self.optimizator = optimizator
        self.gamma = float(gama)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        # Adam state
        self._adam_inicijaliziran = False
        self.t_adam = 0

        # Adagrad state
        self._adagrad_inicijaliziran = False

    def score(self, u: int, i: int, t: int) -> float:
        #vraca \hat{y}_{u,i,t} = sum_{f=1..k} U_u,f * I_i,f * T_t,f
        return float(np.sum(self.U[u] * self.I[i] * self.T[t]))

    def ocijeni_sve_oznake(self, u: int, i: int) -> np.ndarray:
        #za post (u,i)
        ui = self.U[u] * self.I[i]   # (k,)
        return self.T @ ui           # (n_oznake,)

    def _inicijaliz_adam_stanje(self):
        self.mU = np.zeros_like(self.U); self.vU = np.zeros_like(self.U)
        self.mI = np.zeros_like(self.I); self.vI = np.zeros_like(self.I)
        self.mT = np.zeros_like(self.T); self.vT = np.zeros_like(self.T)
        self._adam_inicijaliziran = True
        self.t_adam = 0

    def _inicijaliz_adagrad_stanje(self):
        self.gU = np.zeros_like(self.U)
        self.gI = np.zeros_like(self.I)
        self.gT = np.zeros_like(self.T)
        self._adagrad_inicijaliziran = True

    def fit_bpr(self,
                data: TenzorPodaci,
                steps: int = 200_000,
                lr: float = 0.01,
                reg: float = 1e-5,
                seed: int = 42,
                progress: bool = True) -> "KanonskaDekompozicija":

        sampler = BPRUzorkovanje(data, seed=seed)
        it = range(steps)
        if progress:
            it = tqdm(it, desc="BPR-CD", unit="step")

        for step in it:
            u, i, t_pos, t_neg = sampler.uzorkuj()

            x = self.score(u, i, t_pos) - self.score(u, i, t_neg)
            d = bpr_delta(x)

            #kako se updatea lr
            if self.optimizator == "decay":
                lr_t = lr / (1.0 + self.gamma * step)
            else:
                lr_t = lr

            # cachiram vektore koji mi trebaju za gradijente
            Uu = self.U[u].copy()
            Ii = self.I[i].copy()
            Tpos = self.T[t_pos].copy()
            Tneg = self.T[t_neg].copy()

            # Za PARAFAC:
            # \hat{y}(u,i,t) = sum_{f=1..k} U_u[f]*I_i[f]*T_t[f]
            # parc \hat{y}/U_u = I_i * T_t
            # parc \hat{y}/I_i = U_u * T_t
            # parc \hat{y}/T_t = U_u * I_i
            grad_U = d * (Ii * (Tpos - Tneg)) - reg * Uu
            grad_I = d * (Uu * (Tpos - Tneg)) - reg * Ii
            grad_T_zajednicki = d * (Uu * Ii)  # (k,)

            grad_T_pos = grad_T_zajednicki - reg * Tpos
            grad_T_neg = -grad_T_zajednicki - reg * Tneg

            if self.optimizator == "adam":
                if not self._adam_inicijaliziran:
                    self._inicijaliz_adam_stanje()

                self.t_adam += 1
                t = self.t_adam
                b1, b2 = self.beta1, self.beta2

                # Adam update parametara...
                
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

            elif self.optimizator == "adagrad":
                if not self._adagrad_inicijaliziran:
                    self._inicijaliz_adagrad_stanje()

                self.gU[u] += grad_U**2
                self.gI[i] += grad_I**2
                self.gT[t_pos] += grad_T_pos**2
                self.gT[t_neg] += grad_T_neg**2

                self.U[u] += lr_t * grad_U / (np.sqrt(self.gU[u]) + self.eps)
                self.I[i] += lr_t * grad_I / (np.sqrt(self.gI[i]) + self.eps)
                self.T[t_pos] += lr_t * grad_T_pos / (np.sqrt(self.gT[t_pos]) + self.eps)
                self.T[t_neg] += lr_t * grad_T_neg / (np.sqrt(self.gT[t_neg]) + self.eps)

            else:
                # obični SGD
                self.U[u] += lr_t * grad_U
                self.I[i] += lr_t * grad_I
                self.T[t_pos] += lr_t * grad_T_pos
                self.T[t_neg] += lr_t * grad_T_neg

        return self
