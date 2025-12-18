import pandas as pd
import math
import scipy
import scipy.stats as stats

#C'est la partie la plus importante dans l'analyse de données. D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes. De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste.
#Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées. La statistique univariée est une statistique descriptive. Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant.
#Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide. Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

#Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(nom):
    with open(nom, encoding="utf-8") as fichier:
        contenu = pd.read_csv(fichier)
    return contenu

#Théorie de l'échantillonnage (intervalles de fluctuation)
#L'échantillonnage se base sur la répétitivité.
print("Résultat sur le calcul d'un intervalle de fluctuation")

donnees = pd.DataFrame(ouvrirUnFichier("data/Echantillonnage-100-Echantillons.csv"))

#Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.
print("Résultat sur le calcul d'un intervalle de confiance")

#Théorie de la décision (tests d'hypothèse)
#La décision se base sur la notion de risques alpha et bêta.
#Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Théorie de la décision")

# --- Chargement des 100 échantillons ---
donnees = pd.DataFrame(ouvrirUnFichier("data/Echantillonnage-100-Echantillons.csv"))


moyennes = donnees.mean().round(0)

moy_pour = moyennes["Pour"]
moy_contre = moyennes["Contre"]
moy_sans = moyennes["Sans opinion"]

print("Moyennes observées sur 100 échantillons :")
print(moyennes)


total_moy = moy_pour + moy_contre + moy_sans
freq_obs = {
    "Pour": round(moy_pour / total_moy, 2),
    "Contre": round(moy_contre / total_moy, 2),
    "Sans opinion": round(moy_sans / total_moy, 2)
}

print("\nFréquences observées (moyennes) :")
print(freq_obs)


# Fichier Population réelle

population = pd.DataFrame(ouvrirUnFichier("./data/Echantillonnage-Population-reelle.csv"))

pop_pour = population.iloc[0, 0]
pop_cont = population.iloc[1, 0]
pop_sans = 0

total_pop = pop_pour + pop_cont + pop_sans

freq_pop = {
    "Pour": round(pop_pour / total_pop, 2),
    "Contre": round(pop_cont / total_pop, 2),
    "Sans opinion": round(pop_sans / total_pop, 2)
}

print("\nFréquences de la population réelle :")
print(freq_pop)





n = donnees.iloc[0].sum()     
z = 1.96                       

print("\nIntervalles de fluctuation (95%) :")
for opinion, p_obs in freq_obs.items():
    IF_bas = round(p_obs - z * math.sqrt(p_obs*(1-p_obs)/n), 3)
    IF_haut = round(p_obs + z * math.sqrt(p_obs*(1-p_obs)/n), 3)
    print(f"{opinion} : [{IF_bas} ; {IF_haut}]")


# 2. THÉORIE DE L’ESTIMATION

print("\n=== 2) INTERVALLES DE CONFIANCE ===\n")

# Premier échantillon
ech0 = list(donnees.iloc[0])
total_ech0 = sum(ech0)

freq_ech0 = {
    "Pour": ech0[0]/total_ech0,
    "Contre": ech0[1]/total_ech0,
    "Sans opinion": ech0[2]/total_ech0
}

print("Fréquences du premier échantillon :")
print({k: round(v, 3) for k, v in freq_ech0.items()})

print("\nIntervalles de confiance (95%) :")
for opinion, p_hat in freq_ech0.items():
    IC_bas = round(p_hat - z * math.sqrt(p_hat*(1-p_hat)/total_ech0), 3)
    IC_haut = round(p_hat + z * math.sqrt(p_hat*(1-p_hat)/total_ech0), 3)
    print(f"{opinion} : [{IC_bas} ; {IC_haut}]")


# 3. THÉORIE DE LA DÉCISION

print("\n=== 3) TEST DE NORMALITÉ (Shapiro-Wilk) ===\n")


test1 = pd.read_csv("./data/Loi-normale-Test-1.csv")
test2 = pd.read_csv("./data/Loi-normale-Test-2.csv")


col1 = test1.columns[0]
col2 = test2.columns[0]


stat1, p1 = stats.shapiro(test1[col1])
stat2, p2 = stats.shapiro(test2[col2])

print("Test 1 :")
print(f"  W = {stat1:.4f},  p-value = {p1:.4f}")
print("  → Normal" if p1 > 0.05 else "  → Pas normal")

print("\nTest 2 :")
print(f"  W = {stat2:.4f},  p-value = {p2:.4f}")
print("  → Normal" if p2 > 0.05 else "  → Pas normal")

print("\nFin du programme.")
