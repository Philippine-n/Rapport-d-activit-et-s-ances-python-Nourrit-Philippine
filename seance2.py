#coding:utf8

import os
import pandas as pd
import matplotlib.pyplot as plt

# Dossiers de sortie
IMG_DIR = "./img"
os.makedirs(IMG_DIR, exist_ok=True)

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv","r") as fichier:
    contenu = pd.read_csv(
    "./data/resultats-elections-presidentielles-2022-1er-tour.csv",
    encoding="utf-8-sig"
)



# Question 5 - Afficher un aperçu
print(contenu)

# Question 6 — Nombre de lignes et de colonnes
nb_lignes = len(contenu)
nb_colonnes = contenu.shape[1]
print(f"\nLignes: {nb_lignes} \nColonnes: {nb_colonnes}")

# Question 7 — Type simple par colonne
print("\nTYPES SIMPLES")
def type_simple(s):
    dt = s.dtype
    if pd.api.types.is_integer_dtype(dt): return "int"
    if pd.api.types.is_float_dtype(dt):   return "float"
    if pd.api.types.is_bool_dtype(dt):    return "bool"
    return "str"

for col in contenu.columns:
    print(f"- {col}: {type_simple(contenu[col])}")

# Question 8 — Afficher le nom des colonnes
print("\nNOMS DES COLONNES (via .head())")
print(contenu.head(0))

# Noms de colonnes utiles
col_dept     = "Libellé du département"
col_inscrits = "Inscrits"
col_votants  = "Votants"
col_blancs   = "Blancs"
col_nuls     = "Nuls"
col_exprimes = "Exprimés"
col_abstention = "Abstentions"


# Question 9 — Sélectionner le nombre des inscrits
if col_inscrits in contenu.columns:
    print("\nEXTRAIT Inscrits")
    if col_dept in contenu.columns:
        print(contenu[[col_dept, col_inscrits]].head())
    else:
        print(contenu[col_inscrits].head())

# Question 10 — Sommes des colonnes quantitatives uniquement
print("\n=== SOMMES (colonnes quantitatives) ===")
for col in contenu.select_dtypes(include=["number"]).columns:
    print(f"- {col}: {float(contenu[col].sum(skipna=True))}")

# Question 11 — Diagrammes en barres: Inscrits / Votants par département
if col_dept in contenu.columns and col_inscrits in contenu.columns and col_votants in contenu.columns:
    agg = contenu.groupby(col_dept, dropna=False)[[col_inscrits, col_votants]].sum(numeric_only=True).reset_index()

    colonnes = [col_inscrits, col_votants]

    for col in colonnes:
        plt.figure()
        plt.bar(agg[col_dept].astype(str), agg[col])
        plt.title(f"{col} par département")
        plt.xticks(rotation=90)
        plt.tight_layout()

        # nom du fichier
        filename = f"barres_{col.lower().replace(' ', '_')}_par_departement.png"
        plt.savefig(os.path.join(IMG_DIR, filename), dpi=150)
        plt.close()
        print(f" Diagramme enregistré : {filename}")

# Question 13 — Histogramme de la distribution des 'Inscrits'
if col_inscrits in contenu.columns:
    plt.figure()
    plt.hist(contenu[col_inscrits].dropna().values, bins="auto", density=True)
    plt.title("Distribution des inscrits (histogramme)")
    plt.xlabel("Inscrits")
    plt.ylabel("Densité")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "histogramme_inscrits.png"), dpi=150)
    plt.close()

# Nettoyage pour les noms de fichiers Windows
def safe_filename(name):
    forbidden = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for c in forbidden:
        name = name.replace(c, '_')
    return name


print("\n=== CAMEMBERTS DES VOTES PAR DÉPARTEMENT ===")

if all(col in contenu.columns for col in [col_dept, col_blancs, col_nuls, col_exprimes, col_abstention]):

    agg = contenu.groupby(col_dept, dropna=False)[
        [col_blancs, col_nuls, col_exprimes, col_abstention]
    ].sum(numeric_only=True).reset_index()

    for _, row in agg.iterrows():
        dept = row[col_dept]
        dept_clean = safe_filename(str(dept))

        valeurs = [
            row[col_blancs],
            row[col_nuls],
            row[col_exprimes],
            row[col_abstention]
        ]
        labels = ["Blancs", "Nuls", "Exprimés", "Abstentions"]

        plt.figure()
        plt.pie(valeurs, labels=labels, autopct="%1.1f%%")
        plt.title(f"Répartition des votes – {dept}")

        filename = f"camembert_votes_{dept_clean}.png"
        plt.savefig(os.path.join(IMG_DIR, filename), dpi=150)
        plt.close()

        print(f" Camembert enregistré : {filename}")

else:
    print(" Colonnes nécessaires manquantes.")

