#Import des librairies
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import make_regression


#Import des fichier de données

url_dict = {'2023': "https://www.data.gouv.fr/fr/datasets/r/78348f03-a11c-4a6b-b8db-2acf4fee81b1",
             '2022': "https://www.data.gouv.fr/fr/datasets/r/87038926-fb31-4959-b2ae-7a24321c599a",
             '2021': "https://www.data.gouv.fr/fr/datasets/r/817204ac-2202-4b4a-98e7-4184d154d98c",
             '2020': "https://www.data.gouv.fr/fr/datasets/r/90a98de0-f562-4328-aa16-fe0dd1dca60f",
             '2019': "https://www.data.gouv.fr/fr/datasets/r/3004168d-bec4-44d9-a781-ef16f41856a2",
             }

def get_subsample():
    # Initialise une liste pour stocker les données de toutes les années
    data_all_year = []

    # Itération sur le dictionnaire contenant les années et les URLs des fichiers de données
    for year, url in tqdm(url_dict.items()):
        # Affichage d'un message indiquant le début du traitement pour chaque année
        print(f'----------------start------- {year}-----')

        # Lecture du fichier de données CSV depuis l'URL spécifiée
        data_ = pd.read_csv(url, sep="|")

        # Suppression des colonnes indésirables du DataFrame
        data_ = data_.drop(['Identifiant de document', 'Reference document','1 Articles CGI',
                            '2 Articles CGI', '3 Articles CGI', '4 Articles CGI', '5 Articles CGI',
                            'No disposition','Surface Carrez du 1er lot', '2eme lot', 'Surface Carrez du 2eme lot',
                            '3eme lot', 'Surface Carrez du 3eme lot', '4eme lot',
                            'Surface Carrez du 4eme lot', '5eme lot', 'Surface Carrez du 5eme lot',
                            'Nombre de lots','Prefixe de section', 'Section', 'Identifiant local',
                            'No plan', 'Nature culture','No Volume','1er lot',
                            'Nature culture speciale','B/T/Q', 'Nature mutation', 'Code voie', 'Code departement','Code commune','Surface terrain', 'Code type local' ], axis=1)

        print('suppression indésirable OK')
        print('--------------------------------------------')
        # Renommage de certaines colonnes pour plus de clarté
        data_.rename(columns={'Surface reelle bati': 'Surface'}, inplace=True)
        data_.rename(columns={'Valeur fonciere': 'Prix'}, inplace=True)
        data_.rename(columns={'Nombre pieces principales': 'NPP'}, inplace=True)
        data_.rename(columns={'Type local': 'Type'}, inplace=True)
        print('RENOMMAGE OK')
        print('--------------------------------------------')
        # Filtrage pour ne garder que les types de biens immobiliers "Appartement" ou "Maison"
        data_ = data_[(data_['Type'] == 'Appartement') | (data_['Type'] == 'Maison')]
        print('FILTRAGE OK')
        print('--------------------------------------------')
        # Création d'une colonne 'adresse' en concaténant les colonnes 'No voie', 'Type de voie' et 'Voie'
        data_['Adresse'] = data_.apply(lambda row: f"{row['No voie']} {row['Type de voie']} {row['Voie']}", axis=1)
        print('CREATION ADRESSE  OK')
        print('--------------------------------------------')
        # Suppression des colonnes 'No voie', 'Type de voie' et 'Voie' car elles ont été concaténées dans 'adresse'
        data_.drop(columns=['No voie', 'Type de voie', 'Voie'], inplace=True)
        print('SUPPRESSION SUITE COLONNE ADRESSE  OK')
        print('--------------------------------------------')

        # Remplacement des virgules par des points dans la colonne "Prix" pour assurer le format numérique
        data_['Prix'] = data_['Prix'].str.replace(',', '.')
        print('REMPLACEMENT VIRGULE  OK')
        print('--------------------------------------------')

        # Affichage de la forme du DataFrame après l'importation et le traitement
        print('Shape from import :',data_.shape)

        # Filtrage des données pour ne conserver que celles de la commune de Rouen
        rouen = ['ROUEN']
        data_rouen = data_[data_['Commune'].isin(rouen)]

        # Affichage de la forme du DataFrame après le filtrage par commune
        print('Shape after subsample : ', data_rouen.shape)

        # Ajout d'une colonne 'year' contenant l'année correspondante à chaque entrée
        data_rouen['year'] = year

        # Ajout du DataFrame filtré et traité à la liste des données de toutes les années
        data_all_year.append(data_rouen)

        # Suppression des DataFrames temporaires pour économiser de la mémoire
        del data_
        del data_rouen

        # Affichage d'un message indiquant la fin du traitement pour chaque année
        print(f'--------------YEAR {year} Done -------')

    # Concaténation de tous les DataFrames de chaque année pour former un seul DataFrame global
    print('STARTING CONTENING')
    df = pd.concat(data_all_year)
    print('CONTENING DONE ')

    # Retourne le DataFrame global contenant les données filtrées et traitées de toutes les années
    return df

#Affichage du fichier
df = get_subsample()

import statistics as st  # Importer le module statistics pour les calculs statistiques

def print_separator(character, length=50):
    """Fonction pour imprimer une barre de séparation."""
    print(character * length)

# Convertir les valeurs de la colonne "Prix" et "Surface" en nombres
df['Prix'] = pd.to_numeric(df['Prix'], errors='coerce')
df['Surface'] = pd.to_numeric(df['Surface'], errors='coerce')

# Afficher un message indiquant la fin de la conversion
print('CONVERSION NOMBRE OK')
print_separator('-', 50)

# Supprimer les lignes en doublon dans le DataFrame
df.drop_duplicates(inplace=True)

# Afficher un message indiquant la fin de la suppression des doublons
print('DROP DUPLICATES OK')
print_separator('-', 50)

# Filtrer les données pour exclure les ventes d'appartements en immeuble (prix >= 1 000 000 €)
df = df[df['Prix'] < 1000000]

# Sélectionner les appartements uniquement
appartements = df[df['Type'] == 'Appartement']

# Regrouper les appartements par adresse et compter le nombre d'appartements à chaque adresse
nb_appartements_par_adresse = appartements['Adresse'].value_counts()

# Filtrer les adresses avec plus d'un appartement (plusieurs appartements à la même adresse)
adresses_multiples = nb_appartements_par_adresse[nb_appartements_par_adresse > 1].index

# Afficher un message de vérification des adresses multiples
print("VERIFICATION DES ADRESSES MULTIPLES")
print_separator('-', 50)

# Modifier le prix des appartements à chaque adresse
for adresse in adresses_multiples:
    # Sélectionner les appartements à cette adresse
    appartements_a_cette_adresse = appartements[appartements['Adresse'] == adresse]
    print(f"L'adresse vérifiée est : {adresse}")

    # Calculer le nombre d'appartements à cette adresse
    nombre_appartements = len(appartements_a_cette_adresse)
    print(f"Le nombre d'appartements à cette adresse est : {nombre_appartements}")

    # Afficher les prix initiaux des appartements à cette adresse
    print(f"Les prix initiaux sont : {df.loc[df['Adresse'] == adresse, 'Prix']}")

    # Calculer le nouveau prix en divisant le prix par le nombre d'appartements à cette adresse
    nouveau_prix = df.loc[df['Adresse'] == adresse, 'Prix'] / nombre_appartements
    print(f"Les nouveaux prix sont : {nouveau_prix}")

    # Afficher une barre de séparation
    print_separator('-', 50)

# Afficher le nombre de lignes avec des valeurs nulles dans les colonnes 'Prix', 'Surface' et 'Adresse'
print("Nombre de lignes avec prix vide : {}".format(df['Prix'].isnull().sum()))
print("Nombre de lignes avec surface vide : {}".format(df['Surface'].isnull().sum()))
print("Nombre de lignes avec adresse vide : {}".format(df['Adresse'].isnull().sum()))

print_separator('-', 50)

# Gestion des valeurs manquantes dans la colonne "Prix" en remplaçant par la médiane
median_price = df['Prix'].median()
df['Prix'].fillna(median_price, inplace=True)

# Gestion des valeurs manquantes dans la colonne "Surface" en remplaçant par la médiane
median_surface = df['Surface'].median()
df['Surface'].fillna(median_surface, inplace=True)

# Filtrer les données pour la rive droite (Code postal 76000)
rive_droite = df[df['Code postal'] == 76000]

# Filtrer les données pour la rive gauche (Code postal 76100)
rive_gauche = df[df['Code postal'] == 76100]

# Affichage des statistiques pour la rive droite
print("Rive Droite (Code postal 76000)")
print_separator('-', 50)

# Afficher les statistiques pour les maisons et les appartements de la rive droite
print("Statistiques pour les maisons :")
maisons_rive_droite = rive_droite[rive_droite['Type'] == 'Maison']
print("  - Moyenne des prix : {:.2f} €".format(maisons_rive_droite['Prix'].mean()))
print("  - Médiane des prix : {:.2f} €".format(st.median(maisons_rive_droite['Prix'])))
print("  - Nombre de ventes : {}".format(len(maisons_rive_droite)))

print("Statistiques pour les appartements :")
appartements_rive_droite = rive_droite[rive_droite['Type'] == 'Appartement']
print("  - Moyenne des prix : {:.2f} €".format(appartements_rive_droite['Prix'].mean()))
print("  - Médiane des prix : {:.2f} €".format(st.median(appartements_rive_droite['Prix'])))
print("  - Nombre de ventes : {}".format(len(appartements_rive_droite)))

# Calculer le prix moyen au mètre carré pour la rive droite (76000)
prix_moyen_m2_maisons_rive_droite = maisons_rive_droite['Prix'].mean() / maisons_rive_droite['Surface'].mean()
prix_moyen_m2_appartements_rive_droite = appartements_rive_droite['Prix'].mean() / appartements_rive_droite['Surface'].mean()

print("  - Prix moyen au mètre carré pour les maisons de la rive droite est : {:.2f} €/m²".format(prix_moyen_m2_maisons_rive_droite))
print("  - Prix moyen au mètre carré pour les appartements de la rive droite est : {:.2f} €/m²".format(prix_moyen_m2_appartements_rive_droite))

# Affichage des statistiques pour la rive gauche
print_separator('-', 50)
print("Rive Gauche (Code postal 76100)")
print_separator('-', 50)

# Afficher les statistiques pour les maisons et les appartements de la rive gauche
print("Statistiques pour les maisons :")
maisons_rive_gauche = rive_gauche[rive_gauche['Type'] == 'Maison']
print("  - Moyenne des prix : {:.2f} €".format(maisons_rive_gauche['Prix'].mean()))
print("  - Médiane des prix : {:.2f} €".format(st.median(maisons_rive_gauche['Prix'])))
print("  - Nombre de ventes : {}".format(len(maisons_rive_gauche)))

print("Statistiques pour les appartements :")
appartements_rive_gauche = rive_gauche[rive_gauche['Type'] == 'Appartement']
print("  - Moyenne des prix : {:.2f} €".format(appartements_rive_gauche['Prix'].mean()))
print("  - Médiane des prix : {:.2f} €".format(st.median(appartements_rive_gauche['Prix'])))
print("  - Nombre de ventes : {}".format(len(appartements_rive_gauche)))

# Calculer le prix moyen au mètre carré pour la rive gauche (76100)
prix_moyen_m2_maisons_rive_gauche = maisons_rive_gauche['Prix'].mean() / maisons_rive_gauche['Surface'].mean()
prix_moyen_m2_appartements_rive_gauche = appartements_rive_gauche['Prix'].mean() / appartements_rive_gauche['Surface'].mean()

print("  - Prix moyen au mètre carré pour les maisons de la rive gauche est : {:.2f} €/m²".format(prix_moyen_m2_maisons_rive_gauche))
print("  - Prix moyen au mètre carré pour les appartements de la rive gauche est : {:.2f} €/m²".format(prix_moyen_m2_appartements_rive_gauche))

# Séparation globale pour les résultats finaux
print_separator('-', 50)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Sélection des caractéristiques et de la cible
features = ['Surface', 'Type', 'Code postal', 'NPP']
target = 'Prix'

X = df[features]
y = df[target]

# Encodage des variables catégorielles
categorical_features = ['Type', 'Code postal']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

# Évaluation du modèle
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)


print(f"Score R² sur l'ensemble d'entraînement : {train_score:.2f}")
print(f"Score R² sur l'ensemble de test : {test_score:.2f}")

# Utilisation du modèle pour faire des prédictions
new_data = pd.DataFrame({
    'Surface': [80],
    'Type': ['Appartement'],
    'Code postal': [76000],
    'NPP' : [4]
})

predicted_price = model.predict(new_data)
print(f"Prix prédit pour la nouvelle observation : {predicted_price[0]:.2f} €")
