import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math


# Fonction pour ouvrir les fichiers
def ouvrirUnFichier(nom):
    with open(nom, encoding="utf8") as fichier:
        contenu = pd.read_csv(fichier, sep=",", engine="python")
    return contenu


def conversionLog(liste):
    log = []
    for element in liste:
        try:
            log.append(math.log(float(element)))
        except Exception:
            # si element est Nan ou non convertible, on saute
            continue
    return log

# Trier les listes par ordre décroissant
def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

# le classement des listes spécifiques aux populations
def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        try:
            val = float(pop[element])
            if not math.isnan(val):
                ordrepop.append([val, etat[element]])
        except Exception:
            continue
    ordrepop = sorted(ordrepop, key=lambda x: x[0], reverse=True)
    for idx in range(0, len(ordrepop)):
        ordrepop[idx] = [idx + 1, ordrepop[idx][1]]
    return ordrepop

# Obtenir l'ordre défini entre deux classements
def classementPays(ordre1, ordre2):
    # ordre1 et ordre2 doivent être des listes de la forme [[rang, etat], ...]
    classement = []
    
    d1 = {etat: rang for rang, etat in ordre1}
    d2 = {etat: rang for rang, etat in ordre2}
    
    etats_commun = set(d1.keys()).intersection(set(d2.keys()))
    for etat in etats_commun:
        classement.append([d1[etat], d2[etat], etat])
    return classement



# Partie 1 : la loi rang-taille


# 1) Charger le fichier island-index.csv (le fichier doit se trouver dans src/data/)
iles = pd.DataFrame(ouvrirUnFichier("./data/island-index.csv"))

# 2) Isoler la colonne 'Surface (km2)'
# On caste en liste native Python comme demandé
surface = list(iles["Surface (km²)"])

# 3) Ajouter les surfaces des continents (cast en float)
surface.append(float(85545323))   # Asie / Afrique / Europe
surface.append(float(37856841))   # Amérique
surface.append(float(7768030))    # Antarctique
surface.append(float(7605049))    # Australie

# 4) Ordonner la liste obtenue avec ordreDecroissant()
surface_triee = ordreDecroissant([float(x) for x in surface if not (pd.isna(x))])

# 5) Visualiser la loi rang-taille en créant une image de sortie (linéaire)
plt.figure()
plt.plot(surface_triee)
plt.title("Loi rang-taille : surfaces des iles + continents")
plt.xlabel("Rang")
plt.ylabel("Surface (km2)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./rang_taille_iles.png")   # image de sortie
plt.close()

# 6) Conversion des axes en logarithme
surface_log = conversionLog(surface_triee)
rangs = list(range(1, len(surface_triee) + 1))
rangs_log = conversionLog(rangs)

plt.figure()
plt.plot(rangs_log, surface_log)
plt.title("Loi rang-taille (log-log)")
plt.xlabel("log(rang)")
plt.ylabel("log(surface)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./rang_taille_iles_log.png")
plt.close()

# 7) Reponse (commentaire) :
# On peut tester la relation entre rang et surface en effectuant un ajustement
# linéaire sur les données log-log et en regarder le coef de détermination R^2.
# On peut aussi tester la correlation de rangs (Spearman) entre deux classements
# ou effectuer un test de pente (test t sur la pente).

# Ajustement linéaire sur les log pour estimer l'exposant de la loi rang-taille
if len(rangs_log) >= 2 and len(rangs_log) == len(surface_log):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(rangs_log, surface_log)
    # commentaire insere dans le fichier :
    # Le fit lineaire log-log donne une pente = slope, R^2 = r_value**2
    # On peut donc conclure s'il y a approximativement une loi puissance.



# Partie 2 : population et densité


# 8) Charger le fichier Le-Monde-HS-Etats-du-monde-2007-2025.csv (doit etre dans src/data/)
monde = pd.DataFrame(ouvrirUnFichier("./data/Le-Monde-HS-Etats-du-monde-2007-2025.csv"))

# 9) Isoler les colonnes demandees : 'Etat', 'Pop 2007', 'Pop 2025', 'Densite 2007', 'Densite 2025'
# NOTE: les noms de colonnes peuvent varier selon le CSV (accents / espaces). Adapter si besoin.
# On essaye plusieurs variantes courantes
candidates_etat = ["Etat", "État", "E\u00a0tat", "Etat ", "Pays", "Country"]
candidates_pop2007 = ["Pop 2007", "Pop2007", "Population 2007", "Pop 2007 "]
candidates_pop2025 = ["Pop 2025", "Pop2025", "Population 2025", "Pop 2025 "]
candidates_dens2007 = ["Densite 2007", "Densit\u00e9 2007", "Densite2007"]
candidates_dens2025 = ["Densite 2025", "Densit\u00e9 2025", "Densite2025"]

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: essayer une recherche insensible a la casse
    lower_cols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    raise KeyError(f"Aucune colonne trouv\u00e9e pour {candidates}")

col_etat = find_column(monde, candidates_etat)
col_pop2007 = find_column(monde, candidates_pop2007)
col_pop2025 = find_column(monde, candidates_pop2025)
col_dens2007 = find_column(monde, candidates_dens2007)
col_dens2025 = find_column(monde, candidates_dens2025)

# Extraire les listes natives Python
etats = list(monde[col_etat])
pop2007 = list(monde[col_pop2007])
pop2025 = list(monde[col_pop2025])
dens2007 = list(monde[col_dens2007])
dens2025 = list(monde[col_dens2025])

# 11) Ordonner de maniere decroissante les listes avec ordrePopulation()
classe_pop2007 = ordrePopulation(pop2007, etats)
classe_pop2025 = ordrePopulation(pop2025, etats)
classe_dens2007 = ordrePopulation(dens2007, etats)
classe_dens2025 = ordrePopulation(dens2025, etats)

print(classe_pop2007[:10])

# 12) Utiliser classementPays() pour preparer la comparaison
classement_pop = classementPays(classe_pop2007, classe_pop2025)
classement_dens = classementPays(classe_dens2007, classe_dens2025)

# Ordonner le resultat par rapport au classement 2007 (colonne 0)
classement_pop.sort(key=lambda x: x[0])
classement_dens.sort(key=lambda x: x[0])

# 13) Isoler les deux colonnes sous la forme de listes differentes
# Pour la population : on prend les rangs 2007 et 2025 maintenant alignes par etat
rangs2007_pop = [item[0] for item in classement_pop]
rangs2025_pop = [item[1] for item in classement_pop]

rangs2007_dens = [item[0] for item in classement_dens]
rangs2025_dens = [item[1] for item in classement_dens]

# 14) Calculer Spearman et Kendall
# S'assurer qu'on a au moins 2 elements
if len(rangs2007_pop) >= 2:
    spearman_pop = scipy.stats.spearmanr(rangs2007_pop, rangs2025_pop)
    kendall_pop = scipy.stats.kendalltau(rangs2007_pop, rangs2025_pop)
else:
    spearman_pop = (np.nan, np.nan)
    kendall_pop = (np.nan, np.nan)

if len(rangs2007_dens) >= 2:
    spearman_dens = scipy.stats.spearmanr(rangs2007_dens, rangs2025_dens)
    kendall_dens = scipy.stats.kendalltau(rangs2007_dens, rangs2025_dens)
else:
    spearman_dens = (np.nan, np.nan)
    kendall_dens = (np.nan, np.nan)

# Affichage des resultats dans la console
print("\n=== Correlation Population 2007 vs 2025 ===")
print(f"Spearman : coef = {spearman_pop.statistic:.4f}, p-value = {spearman_pop.pvalue:.2e}")
print(f"Kendall  : tau  = {kendall_pop.statistic:.4f}, p-value = {kendall_pop.pvalue:.2e}")

print("\n=== Correlation Densite 2007 vs 2025 ===")
print(f"Spearman : coef = {spearman_dens.statistic:.4f}, p-value = {spearman_dens.pvalue:.2e}")
print(f"Kendall  : tau  = {kendall_dens.statistic:.4f}, p-value = {kendall_dens.pvalue:.2e}")

# On ecrit egalement les classements dans des CSV pour consultation
pd.DataFrame(classement_pop, columns=["rang_2007", "rang_2025", "etat"]).to_csv("./classement_pop_2007_2025.csv", index=False)
pd.DataFrame(classement_dens, columns=["rang_2007", "rang_2025", "etat"]).to_csv("./classement_dens_2007_2025.csv", index=False)

# Fin du script
