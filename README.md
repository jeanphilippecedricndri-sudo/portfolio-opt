# portopt

Toolkit de recherche en allocation de portefeuille : données, estimateurs, backtest walk-forward, métriques.

```
portopt/
├── data.py         # yfinance -> prix nettoyés -> rendements
├── estimator.py    # 1/N, InverseVolatility, RiskParity (budgets), ERC, HRP + covariances
├── backtester.py   # Backtester(returns, estimator, in_sample, out_of_sample) -> rendements OOS
└── metric.py       # performance, drawdowns, risque de queue, PSR, summary()
tests/              # 37 tests (propriétés théoriques, no look-ahead, maths du backtest)
examples/run_backtest.py
```

## Installation et usage

```bash
pip install -e ".[dev]"
pytest
python examples/run_backtest.py              # univers de 10 ETF multi-actifs via Yahoo
python examples/run_backtest.py --synthetic  # hors ligne
```

```python
from portopt import load_returns, ERC, HRP, EqualWeight, Backtester, run_backtests, summary

R = load_returns(["SPY", "TLT", "GLD", "DBC", "EFA"], start="2008-01-01", cache_dir="data_cache")

bt = Backtester(R, ERC(cov_method="ledoit_wolf"), in_sample=252, out_of_sample=21,
                transaction_cost_bps=5)
oos = bt.run()            # pd.Series des rendements OOS nets de coûts
bt.weights_               # poids cibles à chaque rebalancement
bt.turnover_              # turnover one-way à chaque rebalancement

rets, bts = run_backtests(R, {"1/N": EqualWeight(), "ERC": ERC(), "HRP": HRP()}, 252, 21)
summary(rets, turnover={k: b.annualized_turnover() for k, b in bts.items()})
```

Pour ajouter une méthode : hériter de `BaseEstimator` et implémenter `_compute_weights(returns) -> np.ndarray`. Le backtester l'accepte immédiatement.

---

## 1. Data

Rendement simple et log :

$$R_t = \frac{P_t}{P_{t-1}} - 1, \qquad r_t = \ln\frac{P_t}{P_{t-1}} = \ln(1+R_t)$$

| | agrège en **coupe** (actifs) | agrège dans le **temps** |
|---|---|---|
| simple | $R_p = \sum_i w_i R_i$ exact | $\prod(1+R_t) - 1$ |
| log | faux | $\sum_t r_t$ exact |

**Tips**
* Le backtest et l'optimisation utilisent des rendements **simples** (la linéarité en coupe est ce qui compte pour $w^\top R$).
* `auto_adjust=True` : le `Close` yfinance est ajusté dividendes et splits, donc total return. Calculer des rendements sur des prix non ajustés crée des pertes fictives à chaque détachement.
* Jamais de `bfill` : il recopie le futur dans le passé. On `ffill` avec une limite (jours fériés quand on mélange des places).
* Les NaN initiaux (actif pas encore coté) sont gardés : le backtester n'investit qu'en actifs ayant une fenêtre in-sample complète. Supprimer les actifs "incomplets" en amont crée un biais de survie.

## 2. Estimateurs

### Risk contribution, la brique commune

Volatilité $\sigma_p = \sqrt{w^\top\Sigma w}$. Elle est homogène de degré 1 en $w$, donc par le théorème d'Euler :

$$\sigma_p = \sum_i w_i \frac{\partial \sigma_p}{\partial w_i} = \sum_i \underbrace{w_i \frac{(\Sigma w)_i}{\sigma_p}}_{RC_i}$$

$RC_i$ est la contribution de l'actif $i$ au risque ; $RC_i/\sigma_p$ somme à 1 (`risk_contributions`).

### 1/N

$w_i = 1/N$. Aucune estimation donc aucune erreur d'estimation. DeMiguel, Garlappi & Uppal (2009) : sur la plupart des jeux de données, le mean-variance estimé ne bat pas 1/N out-of-sample, la fenêtre nécessaire pour que l'erreur d'estimation des $\mu$ soit amortie se compte en siècles. C'est **le benchmark à battre**.
Défaut : l'allocation en capital n'est pas une allocation en risque. Sur 60/40 actions/obligations, les actions portent environ 90 % du risque.

### Inverse volatility

$$w_i = \frac{\sigma_i^{-1}}{\sum_j \sigma_j^{-1}}$$

Si toutes les corrélations sont égales ($\rho_{ij}=\rho$), alors $RC_i \propto w_i\sigma_i \cdot(\ldots)$ identique pour tous : **InvVol = ERC exactement** (testé). Avec `exponent=2` (inverse variance) on obtient le min-variance quand $\Sigma$ est diagonale.

### Risk budgeting (`RiskParity`) et ERC

On cherche $w \ge 0$ tel que $RC_i / \sigma_p = b_i$ pour des budgets $b$ donnés. ERC est le cas $b_i = 1/N$.

**Formulation convexe** (Spinu 2013, Roncalli) :

$$y^* = \arg\min_{y>0}\; \tfrac12 y^\top\Sigma y - \sum_i b_i \ln y_i, \qquad w = \frac{y^*}{\mathbf 1^\top y^*}$$

*Démonstration.* Condition du premier ordre : $(\Sigma y)_i = b_i / y_i \Rightarrow y_i(\Sigma y)_i = b_i$. Les risk contributions de $y$ sont donc proportionnelles à $b$, et elles le restent après normalisation car $RC_i(\lambda y) = \lambda^2 RC_i(y)$. En sommant : $y^\top\Sigma y = \sum b_i = 1$. L'objectif est strictement convexe ($\Sigma \succ 0$, $-\ln$ strictement convexe) donc la solution est **unique**. $\blacksquare$

**Solveur** : descente par coordonnées cyclique (Griveau-Billion, Richard & Roncalli 2013). Pour chaque $i$, l'équation $\Sigma_{ii}y_i^2 + c_i y_i - b_i = 0$ avec $c_i = \sum_{j\ne i}\Sigma_{ij}y_j$ a une racine positive unique :

$$y_i \leftarrow \frac{-c_i + \sqrt{c_i^2 + 4\Sigma_{ii} b_i}}{2\Sigma_{ii}}$$

$\Sigma y$ est mis à jour en $O(N)$ par coordonnée, donc $O(N^2)$ par sweep, sans inversion de matrice.

**Propriété clé** (Maillard, Roncalli & Teiletche 2010) :

$$\sigma_{MV} \le \sigma_{ERC} \le \sigma_{1/N}$$

ERC est un compromis entre min-variance (concentré) et 1/N (diversifié en capital). Autre caractérisation : ERC est le min-variance sous contrainte $\sum_i \ln w_i \ge c$.

**Tips**
* ERC n'utilise pas les $\mu$, seulement $\Sigma$, qui s'estime bien mieux (l'erreur sur une moyenne décroît avec la **durée** de l'historique, celle sur une variance avec le **nombre d'observations**, donc la fréquence aide).
* Risk parity surpondère les actifs peu volatils (obligations). En pratique on vise ensuite une vol cible par levier : la performance historique du risk parity doit beaucoup au bull market obligataire 1980-2020. Regarde 2022.
* Les budgets servent à exprimer des vues en risque ("2x plus de risque sur les taux") plutôt qu'en capital.

### HRP (López de Prado 2016)

1. Distance $d_{ij} = \sqrt{\tfrac12(1-\rho_{ij})} \in [0,1]$, une vraie métrique.
2. Clustering hiérarchique (`single` par défaut) puis **quasi-diagonalisation** : on réordonne les actifs selon les feuilles du dendrogramme, les actifs corrélés deviennent voisins.
3. **Bissection récursive** : à chaque split en clusters $L$ et $R$,

$$V_c = \tilde w_c^\top \Sigma_c \tilde w_c,\quad \tilde w_c \propto \operatorname{diag}(\Sigma_c)^{-1}, \qquad \alpha = 1 - \frac{V_L}{V_L+V_R},\quad w_L \mathrel{*}= \alpha,\; w_R \mathrel{*}= 1-\alpha$$

Pourquoi : Markowitz inverse $\Sigma$, et $\operatorname{cond}(\Sigma)$ explose quand les actifs sont corrélés, les petites valeurs propres (bruit) reçoivent les plus gros poids. HRP n'inverse jamais rien : il n'utilise que des diagonales de sous-blocs.

**Tips**
* `split="bisection"` (papier original) coupe la liste triée en deux moitiés **sans respecter le dendrogramme** : un cluster naturel peut être coupé. `split="dendrogram"` suit les vrais splits de l'arbre.
* Le code original de LdP passe la matrice carrée $d$ à `scipy.linkage`, qui la traite alors comme des **observations** et calcule $\tilde d_{ij} = \lVert d_{\cdot i} - d_{\cdot j}\rVert_2$ (distance de distances). Ici on passe la forme condensée, donc la distance $d$ elle-même.
* Avec 2 actifs, HRP = inverse variance (testé). HRP a souvent plus de turnover que ERC car le clustering `single` est instable (effet de chaîne) : essaie `linkage="ward"` ou `"average"`.

### Covariance : `cov_method`

`"sample"`, `"ledoit_wolf"`, `"ewma"` ou un callable `returns -> ndarray`.

Ledoit-Wolf (2004) :

$$\hat\Sigma = \delta\,\mu I + (1-\delta) S,\qquad \mu = \tfrac{\operatorname{tr} S}{N},\qquad \delta^* = \frac{\min(\bar b^2, d^2)}{d^2}$$

avec $d^2 = \lVert S - \mu I\rVert_F^2$ (dispersion de $S$ autour de la cible) et $\bar b^2 = \tfrac1{T^2}\sum_t \lVert x_t x_t^\top - S\rVert_F^2$ (variance d'estimation de $S$). Intuition : plus $S$ est bruitée relativement à sa distance à la cible, plus on shrink. Les valeurs propres extrêmes sont ramenées vers la moyenne, le conditionnement s'améliore (testé).

Règle : quand $N/T$ n'est pas petit (ex. 100 actions sur 252 jours), la covariance empirique est inutilisable pour toute méthode qui l'inverse (Marchenko-Pastur).

## 3. Backtester

```
|<---- in-sample : fit sur r[t-L : t) ---->|<-- hold : r[t : t+H) -->|
                                            t = date de rebalancement
```

* **Pas de look-ahead** : les poids sont calculés avec les rendements jusqu'à $t-1$ et appliqués à partir de $r_t$ (testé avec un estimateur espion).
* **Drift** (`drift=True`) : entre deux rebalancements les poids dérivent avec les prix (buy and hold), c'est ce qu'un portefeuille réel fait :

$$V_k = \sum_i w_i \prod_{s=1}^{k}(1+r_{i,s}),\qquad R_{p,k} = \frac{V_k}{V_{k-1}}-1,\qquad w^{\text{drift}}_i = \frac{w_i \prod_s (1+r_{i,s})}{V_H}$$

  `drift=False` suppose un rebalancement quotidien gratuit vers la cible : c'est plus optimiste que la réalité.
* **Coûts** : turnover one-way $\tau = \sum_i |w^{\text{cible}}_i - w^{\text{drift}}_i|$ mesuré contre les poids **dérivés** (pas contre la cible précédente, erreur fréquente qui sous-estime le turnover). Richesse multipliée par $(1 - c\,\tau)$ au rebalancement.
* **Univers dynamique** : un actif est éligible s'il n'a aucun NaN dans la fenêtre in-sample.
* `window="rolling"` ou `"expanding"`.

**Tips**
* Le choix de $(L, H)$ est lui-même un hyperparamètre : tester plusieurs couples et regarder la **stabilité** du classement des méthodes, pas le meilleur couple (sinon overfitting de backtest).
* Décaler la date de rebalancement (offset de 0 à $H-1$ jours) donne des trajectoires différentes à paramètres égaux : c'est une mesure bon marché du "timing luck".

## 4. Metrics

| Métrique | Formule |
|---|---|
| CAGR | $\left(\prod_t(1+R_t)\right)^{P/T}-1$ |
| Vol annualisée | $\hat\sigma\sqrt{P}$ ($P=252$) |
| Sharpe | $\sqrt P\,\dfrac{\overline{R - r_f}}{\hat\sigma}$ |
| Sortino | $\sqrt P\,\dfrac{\overline{R-\tau}}{\sqrt{\tfrac1T\sum_t \min(R_t-\tau,0)^2}}$ |
| Drawdown | $DD_t = W_t/\max_{s\le t}W_s - 1$ (capital initial compté comme pic) |
| Calmar | $\text{CAGR}/\lvert\text{MaxDD}\rvert$ |
| VaR$_\alpha$ | $-q_\alpha(R)$ |
| CVaR$_\alpha$ | $-\mathbb E[R \mid R \le q_\alpha]$ (cohérente, la VaR ne l'est pas : pas sous-additive) |
| PSR | $\Phi\!\left(\dfrac{(\widehat{SR}-SR^*)\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{SR}+\tfrac{\gamma_4-1}{4}\widehat{SR}^2}}\right)$ |

**Tips**
* **Volatility drag** : $\text{CAGR} \approx \mu - \sigma^2/2$. À moyenne arithmétique égale, la stratégie la moins volatile compose mieux. C'est une des raisons pour lesquelles ERC bat souvent 1/N en CAGR.
* $\sqrt{252}$ n'est valide que si les rendements sont iid. Avec autocorrélation $\rho$ d'ordre 1, le Sharpe annualisé est biaisé (Lo 2002) : typique des actifs illiquides lissés.
* Erreur standard du Sharpe (iid normal) : $\operatorname{se}(\widehat{SR}) \approx \sqrt{(1+\tfrac12\widehat{SR}^2)/T}$. Un Sharpe annuel de 0.5 sur 5 ans a un écart-type d'environ 0.45 : impossible à distinguer de 0. Le PSR corrige en plus de la skewness et de la kurtosis (un Sharpe gonflé par du short-vol a $\gamma_3<0$, $\gamma_4\gg3$, donc un PSR plus faible).
* Si tu as testé $K$ variantes, compare le meilleur Sharpe au **Deflated Sharpe Ratio** (Bailey & LdP 2014) qui remplace $SR^*$ par le maximum attendu de $K$ Sharpes nuls.

## Références

* DeMiguel, Garlappi, Uppal (2009), *Optimal Versus Naive Diversification*, RFS.
* Maillard, Roncalli, Teiletche (2010), *The Properties of Equally Weighted Risk Contribution Portfolios*, JPM.
* Griveau-Billion, Richard, Roncalli (2013), *A Fast Algorithm for Computing High-dimensional Risk Parity Portfolios*.
* Spinu (2013), *An Algorithm for Computing Risk Parity Weights*.
* López de Prado (2016), *Building Diversified Portfolios that Outperform Out of Sample*, JPM.
* Ledoit, Wolf (2004), *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*, JMVA.
* Bailey, López de Prado (2012), *The Sharpe Ratio Efficient Frontier*, JoR.
* Roncalli (2013), *Introduction to Risk Parity and Budgeting*, CRC.
