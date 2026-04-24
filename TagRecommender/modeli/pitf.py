#PITF/ Pairwise interaction tensor factorization model za BPR


from __future__ import annotations

import numpy as np
from tqdm import tqdm

from TagRecommender.bazniModel import BazniOznakaModel, bpr_delta
from TagRecommender.data import TenzorPodaci
from TagRecommender.uzorkovanje import BPRUzorkovanje


class PITF(BazniOznakaModel):
    """
    PITF model (jednadžba (13)):
        y(u,i,t) = <U[u], TU[t]> + <I[i], TI[t]>
    Parametri:
        U  : (n_korisnici, k)
        I  : (n_artikli, k)
        TU : (n_oznake,  k)
        TI : (n_oznake,  k)
    """
    def __init__(self, n_korisnici: int,
                 n_artikli: int,
                 n_oznake: int,
                 k: int,
                 init_std: float = 0.01,
                 seed: int = 42,
                 optimizator: str | None = None,
                 gama: float = 1e-6,
                 beta1: float = 0.9, #predlozeni pocetni parametri za Adagrad u originalnom radu
                 beta2: float = 0.999,
                 eps: float = 1e-8):
        if optimizator not in (None, "decay", "adam", "adagrad"):
            raise ValueError("optimizator mora biti None, 'decay', 'adam' ili 'adagrad'")

        rng = np.random.default_rng(seed)
        self.k = int(k)

        self.U  = rng.normal(0.0, init_std, size=(n_korisnici, k)).astype(np.float64)
        self.I  = rng.normal(0.0, init_std, size=(n_artikli, k)).astype(np.float64)
        self.TU = rng.normal(0.0, init_std, size=(n_oznake,  k)).astype(np.float64)
        self.TI = rng.normal(0.0, init_std, size=(n_oznake,  k)).astype(np.float64)

        # hiperparametri optimizatora
        self.optimizator = optimizator
        self.gamma = float(gama)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        # inicijalno stanje adama
        self._adam_inicijaliziran = False
        self.t_adam = 0

        # inicijalno stanje adagarda
        self._adagrad_inicijaliziran = False

    def score(self, u: int, i: int, t: int) -> float:
        #vraca \hat{y}_{u,i,t} = <U_u, TU_t> + <I_i, TI_t>
        return float(self.U[u] @ self.TU[t] + self.I[i] @ self.TI[t])

    def ocijeni_sve_oznake(self, u: int, i: int) -> np.ndarray:
        # (n_oznake,)
        return self.TU @ self.U[u] + self.TI @ self.I[i]

    def _inicijaliz_adam_stanje(self):
        #inicijaliziram na nulu
        self.mU  = np.zeros_like(self.U);  self.vU  = np.zeros_like(self.U)
        self.mI  = np.zeros_like(self.I);  self.vI  = np.zeros_like(self.I)
        self.mTU = np.zeros_like(self.TU); self.vTU = np.zeros_like(self.TU)
        self.mTI = np.zeros_like(self.TI); self.vTI = np.zeros_like(self.TI)
        self._adam_inicijaliziran = True
        self.t_adam = 0

    def _inicijaliz_adagrad_stanje(self):
        #G_0 - nule
        self.gU  = np.zeros_like(self.U)
        self.gI  = np.zeros_like(self.I)
        self.gTU = np.zeros_like(self.TU)
        self.gTI = np.zeros_like(self.TI)
        self._adagrad_inicijaliziran = True
    
    def fit_bpr(self, data: TenzorPodaci, steps: int = 200_000,
                lr: float = 0.05, reg: float = 5e-5,
                seed: int = 42, progress: bool = True) -> "PITF":
        sampler = BPRUzorkovanje(data, seed=seed)
        it = range(steps)
        if progress:
            it = tqdm(it, desc="BPR-PITF", unit="step")

        for step in it:
            #optimizacija PITF modela s LearnBPR
            u, i, t_pos, t_neg = sampler.uzorkuj()

            x = self.score(u, i, t_pos) - self.score(u, i, t_neg)
            d = bpr_delta(x)
            
            if self.optimizator is None or self.optimizator in ("adam", "adagrad"):
                lr_t = lr
            elif self.optimizator == "decay":
                lr_t = lr / (1.0 + self.gamma * step)
            else:
                lr_t = lr  # defaultni slucaj, learning rate je konstantan

            #cachiram potrebne vektore za izracun
            Uu = self.U[u].copy()
            Ii = self.I[i].copy()
    
            TU_pos = self.TU[t_pos].copy()
            TU_neg = self.TU[t_neg].copy()
    
            TI_pos = self.TI[t_pos].copy()
            TI_neg = self.TI[t_neg].copy()

                
            grad_U      = d * (TU_pos - TU_neg) - reg * Uu
            grad_I      = d * (TI_pos - TI_neg) - reg * Ii

            grad_TU_pos = d * Uu - reg * TU_pos
            grad_TU_neg = -d * Uu - reg * TU_neg

            grad_TI_pos = d * Ii - reg * TI_pos
            grad_TI_neg = -d * Ii - reg * TI_neg

            if self.optimizator == "adam":
                #adam je otporniji na šum od običnog SGD - parametri s velikom varijancom dobiju manje korake, s manjom veće korake
                #otporan na nagle promjene (momentum - prosjek gradijenata, umjesto da samo gleda prošli)
                if not self._adam_inicijaliziran:
                    self._inicijaliz_adam_stanje()
                self.t_adam += 1
                b1, b2 = self.beta1, self.beta2
                t = self.t_adam

                #iteracije su ovog oblika:
                #računam prosjek dosadašnjih gradijenata:
                #m_t=beta1 * m_{t−1} + (1−beta1) * gradijent_t
                #i prosjek kvadrata gradijenata:
                #(eksponencijalno "zaboravlja" stare gradijente)
                #v_t=beta2 * v_{t-1} + (1-beta2) * gradijent_t**2

                #uklanjam pristranost (dobivena početnim postavljanjem)
                #https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for

                #\hat{m_t}=m_t / (1 - beta1**t)
                #\hat{v_t}=v_t / (1 - beta2**t)

                #i konačno update
                #theta = theta + lr * (\hat{m_t})/(sqrt(\hat{v_t})+epsilon)

                #ovo primjenim na svih 6 jednadžbi iz Figure 5. iz rada
                
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

                # TU[t_pos], TU[t_neg]
                self.mTU[t_pos] = b1*self.mTU[t_pos] + (1-b1)*grad_TU_pos
                self.vTU[t_pos] = b2*self.vTU[t_pos] + (1-b2)*(grad_TU_pos**2)
                m_hat = self.mTU[t_pos] / (1 - b1**t)
                v_hat = self.vTU[t_pos] / (1 - b2**t)
                self.TU[t_pos] += lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

                self.mTU[t_neg] = b1*self.mTU[t_neg] + (1-b1)*grad_TU_neg
                self.vTU[t_neg] = b2*self.vTU[t_neg] + (1-b2)*(grad_TU_neg**2)
                m_hat = self.mTU[t_neg] / (1 - b1**t)
                v_hat = self.vTU[t_neg] / (1 - b2**t)
                self.TU[t_neg] += lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

                # TI[t_pos], TI[t_neg]
                self.mTI[t_pos] = b1*self.mTI[t_pos] + (1-b1)*grad_TI_pos
                self.vTI[t_pos] = b2*self.vTI[t_pos] + (1-b2)*(grad_TI_pos**2)
                m_hat = self.mTI[t_pos] / (1 - b1**t)
                v_hat = self.vTI[t_pos] / (1 - b2**t)
                self.TI[t_pos] += lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

                self.mTI[t_neg] = b1*self.mTI[t_neg] + (1-b1)*grad_TI_neg
                self.vTI[t_neg] = b2*self.vTI[t_neg] + (1-b2)*(grad_TI_neg**2)
                m_hat = self.mTI[t_neg] / (1 - b1**t)
                v_hat = self.vTI[t_neg] / (1 - b2**t)
                self.TI[t_neg] += lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

            elif self.optimizator == "adagrad":
                #svaki parametar ima svoj learning rate!
                #parametri koji često imaju velike gradijente dobit će manji learning rate
                #oni koji imaju male gradijente i rijetko se pojave dobit će veći learning rate
                #ovo je dobro za rijetke podatke, kao što su naši tenzori
                if not self._adagrad_inicijaliziran:
                    self._inicijaliz_adagrad_stanje()

                #G_t = G_{t-1} + gradijent_t**2
                #svaki latentni faktor dobije svoj G
                self.gU[u]      += grad_U**2
                self.gI[i]      += grad_I**2
                self.gTU[t_pos] += grad_TU_pos**2
                self.gTU[t_neg] += grad_TU_neg**2
                self.gTI[t_pos] += grad_TI_pos**2
                self.gTI[t_neg] += grad_TI_neg**2

                #sad Adagrad update:
                # theta_t = theta_{t-1} + lr / (sqrt(G_t) + epsilon) * gradijent_t
                self.U[u]      += lr_t * grad_U      / (np.sqrt(self.gU[u])      + self.eps)
                self.I[i]      += lr_t * grad_I      / (np.sqrt(self.gI[i])      + self.eps)
                self.TU[t_pos] += lr_t * grad_TU_pos / (np.sqrt(self.gTU[t_pos]) + self.eps)
                self.TU[t_neg] += lr_t * grad_TU_neg / (np.sqrt(self.gTU[t_neg]) + self.eps)
                self.TI[t_pos] += lr_t * grad_TI_pos / (np.sqrt(self.gTI[t_pos]) + self.eps)
                self.TI[t_neg] += lr_t * grad_TI_neg / (np.sqrt(self.gTI[t_neg]) + self.eps)

            else:
                # običan SGD (sa ili bez alfa decay preko lr_t)
                #theta = theta + lr_t * gradijent
                #lr_t ili konstantan ili lr_t=lr/(1+gama*korak)
                self.U[u]      += lr_t * grad_U
                self.I[i]      += lr_t * grad_I
                self.TU[t_pos] += lr_t * grad_TU_pos
                self.TU[t_neg] += lr_t * grad_TU_neg
                self.TI[t_pos] += lr_t * grad_TI_pos
                self.TI[t_neg] += lr_t * grad_TI_neg            
            
        return self
