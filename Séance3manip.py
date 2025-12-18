import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

# Sources des données : production de M. Forriez, 2016-2023

# ouvrir le fichier csv
csv_path=os.path.join("data","resultats-elections-presidentielles-2022-1er-tour.csv")

try:
    df = pd.read_csv(csv_path,low_memory=False)
    print ("fichier lu:",csv_path)
except Exception as e:
    print("erreur lecture fichier:",csv_path)
    raise

# afficher une petite partie du tableau
print("\naperçu des 5 premieres lignes:")
print(df.head().to_string(index=False))

# question 5
quant_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("colonnes quantitatives")
print(quant_cols)

# question 5 (suite)
moyennes = df[quant_cols].mean().round(2)
medians = df[quant_cols].median().round(2)
modes_df = df[quant_cols].mode()
modes = modes_df.iloc[0].round(2) if not modes_df.empty else pd.Series([np.nan]*len(quant_cols), index=quant_cols)
stds = df[quant_cols].std().round(2)
abs_dev_mean = df[quant_cols].apply(lambda x: np.mean(np.abs(x - x.mean()))).round(2)
etendue = (df[quant_cols].max() - df[quant_cols].min()).round(2)

# Question 6
stats = pd.DataFrame({
    "Moyenne": moyennes,
    "Médiane": medians,
    "Mode (1er)": modes,
    "Écart-type": stds,
    "Écart absolu moyen": abs_dev_mean,
    "Étendue": etendue
})

print("\n--- Résumé des paramètres statistiques ---")
print(stats)


# Question 7

iqr = (df[quant_cols].quantile(0.75) - df[quant_cols].quantile(0.25)).round(2)
idr = (df[quant_cols].quantile(0.9) - df[quant_cols].quantile(0.1)).round(2)

print("\n--- Distance interquartile (IQR) ---")
print(iqr)

print("\n--- Distance interdécile (IDR) ---")
print(idr)


# Question 8


os.makedirs("img", exist_ok=True)

print("\nGénération des boxplots dans le dossier img/...")

for col in quant_cols:
    plt.figure() 
    df.boxplot(column=[col])
    plt.title(f"Boxplot de {col}")
    plt.savefig(f"img/boxplot_{col}.png")
    plt.close()


# Question 9

island_path = os.path.join("data", "island-index.csv")

try:
    island_df = pd.read_csv(island_path)
    print("\nfichier lu :", island_path)
except Exception as e:
    print("Erreur lors de la lecture de", island_path)
    raise

print("\nAperçu du fichier island-index :")
print(island_df.head().to_string(index=False))


# Question 10


surface = island_df["Surface (km²)"]


classes = {
    "0-10": surface[(surface > 0) & (surface <= 10)],
    "10-25": surface[(surface > 10) & (surface <= 25)],
    "25-50": surface[(surface > 25) & (surface <= 50)],
    "50-100": surface[(surface > 50) & (surface <= 100)],
    "100-250": surface[(surface > 100) & (surface <= 250)],
    "250-500": surface[(surface > 250) & (surface <= 500)],
    "500-1000": surface[(surface > 500) & (surface <= 1000)],
    "1000+": surface[(surface > 1000)]
}

print("\n--- Nombre d'îles par catégorie ---")
for cat, values in classes.items():
    print(f"{cat} km2 : {len(values)} îles")
