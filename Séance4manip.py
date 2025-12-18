import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt

#https://docs.scipy.org/doc/scipy/reference/stats.html


dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print(dist_names)

# loi de dirac
def visualiser_dirac(valeur=0):
    """
    Affiche la "loi de Dirac" discrète centrée sur 'valeur' (toutes les prob = 0 sauf pour 'valeur').
    C'est une représentation pédagogique (impulsion).
    """
    x = np.arange(valeur - 3, valeur + 4)
    y = np.array([1.0 if xi == valeur else 0.0 for xi in x])
    plt.bar(x, y, edgecolor='black')
    plt.title(f"Loi de Dirac centrée sur {valeur}")
    plt.xlabel("x")
    plt.ylabel("Probabilité")
    plt.ylim(0, 1.1)
    plt.show()

# Loi uniforme discrete
def visualiser_uniforme_discrete(a, b, taille_simulation=10000):
    """
    Loi uniforme discrète sur {a, ..., b} :
    - affiche la PMF théorique
    - calcule moyenne et écart-type théoriques
    - simule 'taille_simulation' tirages et compare moyennes/écarts-types empiriques
    """
    if b < a:
        raise ValueError("b doit être >= a")
    valeurs = np.arange(a, b+1)
    n = len(valeurs)
    proba = np.full(n, 1.0 / n)


    moyenne_theo = (a + b) / 2.0
    var_theo = (n**2 - 1) / 12.0
    ecart_type_theo = np.sqrt(var_theo)

    print(f"Uniforme discrète sur {{{a},...,{b}}}")
    print(f"  Moyenne théorique : {moyenne_theo:.4f}")
    print(f"  Écart-type théorique : {ecart_type_theo:.4f}")

    
    echantillon = np.random.randint(a, b + 1, size=taille_simulation)
    moy_emp = echantillon.mean()
    std_emp = echantillon.std(ddof=0)
    print(f"  Simulation ({taille_simulation} tirages) -> moyenne emp = {moy_emp:.4f}, std emp = {std_emp:.4f}")

    
    plt.bar(valeurs, proba, edgecolor='black')
    plt.title(f"PMF - uniforme discrète {{{a}..{b}}}")
    plt.xlabel("Valeurs")
    plt.ylabel("Probabilité")
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.show()

# loi binominale
def visualiser_binomiale(n, p, taille_simulation=10000):
    """
    Loi binomiale B(n, p). Affiche la PMF théorique, moyenne/std théoriques,
    et résultats empiriques par simulation.
    """
    k = np.arange(0, n+1)
    pmf = stats.binom.pmf(k, n, p)
    moyenne_theo = n * p
    std_theo = np.sqrt(n * p * (1 - p))

    print(f"Binomiale B(n={n}, p={p})")
    print(f"  Moyenne théorique : {moyenne_theo:.4f}")
    print(f"  Écart-type théorique : {std_theo:.4f}")

    
    echantillon = stats.binom.rvs(n, p, size=taille_simulation)
    print(f"  Simulation ({taille_simulation}) -> moyenne emp = {echantillon.mean():.4f}, std emp = {echantillon.std(ddof=0):.4f}")

    plt.bar(k, pmf, edgecolor='black')
    plt.title(f"PMF - Binomiale B({n},{p})")
    plt.xlabel("k")
    plt.ylabel("P(X=k)")
    plt.show()

# Loi de Poisson
def visualiser_poisson(lmbda, max_k=None, taille_simulation=10000):
    """
    Loi de Poisson(λ). Affiche la PMF, théorie et simulation.
    """
    if max_k is None:
        max_k = max(15, int(4 * lmbda))
    k = np.arange(0, max_k + 1)
    pmf = stats.poisson.pmf(k, lmbda)
    print(f"Poisson(λ={lmbda}) — Moyenne théorique = {lmbda:.4f}, écart-type théorique = {np.sqrt(lmbda):.4f}")

    echantillon = stats.poisson.rvs(lmbda, size=taille_simulation)
    print(f"  Simulation ({taille_simulation}) -> moyenne emp = {echantillon.mean():.4f}, std emp = {echantillon.std(ddof=0):.4f}")

    plt.bar(k, pmf, edgecolor='black')
    plt.title(f"PMF - Poisson(λ={lmbda})")
    plt.xlabel("k")
    plt.ylabel("P(X=k)")
    plt.show()

# loi Zipf-Mandelbrot
def zipf_mandelbrot_pmf(s, q, N):
    """
    PMF non normalisée de Zipf-Mandelbrot: p(k) ∝ 1/(k+q)^s pour k=1..N.
    On normalise sur 1..N.
    s > 0, q >= 0
    """
    k = np.arange(1, N+1)
    weights = 1.0 / np.power(k + q, s)
    pmf = weights / weights.sum()
    return k, pmf


def visualiser_zipf_mandelbrot(s=1.1, q=0.0, N=50, taille_simulation=10000):
    """
    Affiche la loi Zipf-Mandelbrot sur k=1..N et compare la moyenne théorique (approx) et simulation.
    """
    k, pmf = zipf_mandelbrot_pmf(s, q, N)
    
    moyenne_theo = (k * pmf).sum()
    var_theo = ((k - moyenne_theo)**2 * pmf).sum()
    std_theo = np.sqrt(var_theo)

    print(f"Zipf-Mandelbrot (s={s}, q={q}, N={N})")
    print(f"  Moyenne (sur 1..N) = {moyenne_theo:.4f}, écart-type = {std_theo:.4f}")

    
    echantillon = np.random.choice(k, size=taille_simulation, p=pmf)
    print(f"  Simulation ({taille_simulation}) -> moyenne emp = {echantillon.mean():.4f}, std emp = {echantillon.std(ddof=0):.4f}")

    plt.bar(k, pmf, edgecolor='black')
    plt.title(f"Zipf-Mandelbrot s={s}, q={q} (N={N})")
    plt.xlabel("k")
    plt.ylabel("P(k)")
    plt.show()



# Distributions statistiques de variables continues:

# Loi normale
def visualiser_normale(mu=0, sigma=1, taille_simulation=10000):
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    print(f"Loi normale N({mu},{sigma**2}) — moyenne théorique = {mu}, écart-type = {sigma}")
    echantillon = stats.norm.rvs(loc=mu, scale=sigma, size=taille_simulation)
    print(f"  Simulation ({taille_simulation}) -> moyenne emp = {echantillon.mean():.4f}, std emp = {echantillon.std(ddof=0):.4f}")
    plt.plot(x, pdf)
    plt.title(f"PDF - Normale N({mu},{sigma**2})")
    plt.xlabel("x")
    plt.ylabel("densité")
    plt.show()

# Loi log-normale
def visualiser_lognormale(s=1.0, scale=1.0, taille_simulation=10000):
    """
    scipy parametrise lognorm via 's' (shape) et 'scale' (exp(mu)).
    On affiche la PDF et on simule.
    """
    x = np.linspace(0.001, 10 * scale, 400)
    pdf = stats.lognorm.pdf(x, s, scale=scale)
    mean_theo = stats.lognorm.mean(s, scale=scale)
    std_theo = stats.lognorm.std(s, scale=scale)
    print(f"Loi log-normale (s={s}, scale={scale}) -> moyenne théorique ≈ {mean_theo:.4f}, std théorique ≈ {std_theo:.4f}")
    echantillon = stats.lognorm.rvs(s, scale=scale, size=taille_simulation)
    print(f"  Simulation ({taille_simulation}) -> moyenne emp = {echantillon.mean():.4f}, std emp = {echantillon.std(ddof=0):.4f}")
    plt.plot(x, pdf)
    plt.title("PDF - Log-Normale")
    plt.xlabel("x")
    plt.ylabel("densité")
    plt.show()

# Loi uniforme
def visualiser_uniforme_continue(a=0.0, b=1.0, taille_simulation=10000):
    x = np.linspace(a, b, 200)
    pdf = stats.uniform.pdf(x, loc=a, scale=b - a)
    mean_theo = (a + b) / 2.0
    std_theo = np.sqrt((b - a)**2 / 12.0)
    print(f"Uniforme continue U({a},{b}) -> moyenne = {mean_theo:.4f}, std = {std_theo:.4f}")
    echantillon = stats.uniform.rvs(loc=a, scale=b - a, size=taille_simulation)
    print(f"  Simulation ({taille_simulation}) -> moyenne emp = {echantillon.mean():.4f}, std emp = {echantillon.std(ddof=0):.4f}")
    plt.plot(x, pdf)
    plt.title(f"PDF - Uniforme continue U({a},{b})")
    plt.xlabel("x")
    plt.ylabel("densité")
    plt.show()

# Loi X2 
def visualiser_chi2(df=3, taille_simulation=10000):
    x = np.linspace(0, df + 10, 400)
    pdf = stats.chi2.pdf(x, df)
    mean_theo = df
    std_theo = np.sqrt(2 * df)
    print(f"Chi2(df={df}) -> moyenne = {mean_theo:.4f}, std = {std_theo:.4f}")
    echantillon = stats.chi2.rvs(df, size=taille_simulation)
    print(f"  Simulation ({taille_simulation}) -> moyenne emp = {echantillon.mean():.4f}, std emp = {echantillon.std(ddof=0):.4f}")
    plt.plot(x, pdf)
    plt.title(f"PDF - Chi2 (df={df})")
    plt.xlabel("x")
    plt.ylabel("densité")
    plt.show()

# Loi de Paretoreto
def visualiser_pareto(b=2.0, scale=1.0, taille_simulation=10000):
    """
    scipy.stats.pareto: paramètre 'b' (shape) et 'scale' (scale).
    La distribution de Pareto (type I) a queue lourde.
    """
    x = np.linspace(scale, scale * 10, 400)
    pdf = stats.pareto.pdf(x, b, scale=scale)
    
    try:
        mean_theo = stats.pareto.mean(b, scale=scale)
        std_theo = stats.pareto.std(b, scale=scale)
        print(f"Pareto(b={b}, scale={scale}) -> moyenne théorique = {mean_theo:.4f}, std théorique = {std_theo:.4f}")
    except Exception:
        print("Pareto: moyenne ou std théorique non définie (paramètres).")
    echantillon = stats.pareto.rvs(b, scale=scale, size=taille_simulation)
    print(f"  Simulation ({taille_simulation}) -> moyenne emp = {echantillon.mean():.4f}, std emp = {echantillon.std(ddof=0):.4f}")
    plt.plot(x, pdf)
    plt.title(f"PDF - Pareto (b={b})")
    plt.xlabel("x")
    plt.ylabel("densité")
    plt.yscale('log')  
    plt.show()



def exemples():
    # Distributions statistiques de variables discretes
    visualiser_dirac(2)
    visualiser_uniforme_discrete(1, 6) 
    visualiser_binomiale(n=10, p=0.3)
    visualiser_poisson(lmbda=3)
    visualiser_zipf_mandelbrot(s=1.1, q=0.0, N=50)

    # Distributions statistiques de variables continues
    visualiser_normale(mu=0, sigma=1)
    visualiser_lognormale(s=0.9, scale=1.0)
    visualiser_uniforme_continue(a=0, b=1)
    visualiser_chi2(df=4)
    visualiser_pareto(b=2.5, scale=1.0)


if __name__ == "__main__":
    print("Exemples pour la séance 4 — visualisations et simulations")
    exemples()
