import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from io import StringIO
from datetime import datetime
import glob
import os
import plotly.graph_objects as go
import math

# === Fonctions de calcul de CV ===
def cv(x):
    x = pd.to_numeric(x, errors='coerce')
    m = np.nanmean(x)
    return np.nan if m == 0 or np.isnan(m) else (np.nanstd(x) / m) * 100

def sd(x):
    x = pd.to_numeric(x, errors='coerce')
    sd = np.nanstd(x)
    return sd

def cv_robuste_iqr(x):
    x = pd.to_numeric(x, errors='coerce')
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    return np.nan if med == 0 or np.isnan(med) else (iqr / med) * 100

def sd_robuste_iqr(x):
    x = pd.to_numeric(x, errors='coerce')
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    return iqr

def cv_robuste_iqr2(x):
    x = pd.to_numeric(x, errors='coerce')
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    sigma_robuste = iqr / 1.349
    return np.nan if med == 0 or np.isnan(med) else (sigma_robuste / med) * 100

def sd_robuste_iqr2(x):
    x = pd.to_numeric(x, errors='coerce')
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    sigma_robuste = iqr / 1.349
    return sigma_robuste

def cv_robuste_mad(x):
    x = pd.to_numeric(x, errors='coerce')
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return np.nan if med == 0 or np.isnan(med) else (mad / med) * 100

def mad(x):
    x = pd.to_numeric(x, errors='coerce')
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med)) * 1.4826

def trouver_lot_niveau_proche(row, ciq_moyennes):
    # Filtrer sur les clés, par exemple Nickname, Paramètre, Annee
    filtres = (
        (ciq_moyennes["Nickname"] == row["Nickname"]) &
        (ciq_moyennes["Paramètre"] == row["Paramètre"]) &
        (ciq_moyennes["Annee"] == row["Annee"])
    )
    candidats = ciq_moyennes.loc[filtres].copy()

    # Si pas de correspondance, retourner NaN
    if candidats.empty:
        return np.nan

    resultat = row.get("Resultat", np.nan)
    if pd.isna(resultat):
        return np.nan

    # S'assurer que la colonne de comparaison est numérique
    candidats["moy_valeur"] = pd.to_numeric(candidats["moy_valeur"], errors="coerce")
    candidats = candidats.dropna(subset=["moy_valeur"])

    if candidats.empty:
        return np.nan

    # Calculer la différence absolue et trouver la plus proche
    candidats["ecart"] = (candidats["moy_valeur"] - resultat).abs()
    idx_min = candidats["ecart"].idxmin()

    return candidats.loc[idx_min, "lot_niveau"]

def lire_CIQ_csv(fichier_path=None, contenu_brut=None, nom=""):
    """
    Traite un fichier CSV soit à partir d’un chemin local (fichier_path),
    soit à partir d’un contenu brut (contenu_brut).
    Ignore la première ligne, utilise ',' comme séparateur.
    """
    try:
        if fichier_path:
            with open(fichier_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read().splitlines()
        elif contenu_brut:
            content = contenu_brut.decode('utf-8', errors='replace').splitlines()
        else:
            return None

        if len(content) < 2:
            st.warning(f"Le fichier {nom} semble vide ou mal formaté.")
            return None

        lines = content[1:]  # ignorer la première ligne
        content_str = StringIO('\n'.join(lines))

        df = pd.read_csv(content_str, sep=',', on_bad_lines='skip')
        return df

    except Exception as e:
        st.error(f"Erreur lecture du fichier {nom or fichier_path} : {e}")
        return None

def nettoyer_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'HGB(g/dL)' in df.columns and 'HGB(g/L)' not in df.columns:
        df['HGB(g/dL)'] = pd.to_numeric(df['HGB(g/dL)'], errors='coerce') * 10
        df.rename(columns={'HGB(g/dL)': 'HGB(g/L)'}, inplace=True)

    if 'MCHC(g/dL)' in df.columns and 'MCHC(g/L)' not in df.columns:
        df['MCHC(g/dL)'] = pd.to_numeric(df['MCHC(g/dL)'], errors='coerce') * 10
        df.rename(columns={'MCHC(g/dL)': 'MCHC(g/L)'}, inplace=True)

    df.rename(columns=lambda col: col.replace('(10^3/uL)', '(10^9/L)') if '(10^3/uL)' in col else col, inplace=True)
    df.rename(columns=lambda col: col.replace('(10^6/uL)', '(10^12/L)') if '(10^6/uL)' in col else col, inplace=True)

    return df


st.title("Analyse des coefficients de variation (CV)")

# === Choix de la source de données ===
choix_source = st.radio(
    "Choisissez la source des données :",
    ["Importer des fichiers CSV", "Utiliser les données par défaut", "Rechercher un fichier lot*.csv localement"]
)

if choix_source == "Importer des fichiers CSV":
    uploaded_files = st.file_uploader("Importer un ou plusieurs fichiers CSV", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        list_df = []
        for file in uploaded_files:
            df = lire_CIQ_csv(contenu_brut=file.read(), nom=file.name)
            if df is not None:
                df = nettoyer_colonnes(df)
                list_df.append(df)

        if list_df:
            CIQ = pd.concat(list_df, ignore_index=True)
            colonnes_dupliquees = CIQ.columns[CIQ.columns.duplicated()].tolist()
            if colonnes_dupliquees:
                st.error(f"Doublons de colonnes détectés : {colonnes_dupliquees}")
                st.stop()
            st.success(f"{len(list_df)} fichier(s) chargé(s), total : {CIQ.shape[0]} lignes.")
        else:
            st.warning("Aucun fichier n'a pu être chargé correctement.")
            st.stop()
    else:
        st.stop()

elif choix_source == "Utiliser les données par défaut":
    df = lire_CIQ_csv(fichier_path="lot_default.csv")
    if df is not None:
        df = nettoyer_colonnes(df)
        colonnes_dupliquees = df.columns[df.columns.duplicated()].tolist()
        if colonnes_dupliquees:
            st.error(f"Doublons de colonnes détectés : {colonnes_dupliquees}")
            st.stop()
        CIQ = df
        st.success("Données par défaut chargées depuis `lot_default.csv`.")
    else:
        st.warning("Impossible de charger `lot_default.csv`.")
        st.stop()

elif choix_source == "Rechercher un fichier lot*.csv localement":
    fichiers = glob.glob("lot*.csv")
    if fichiers:
        fichiers_selectionnes = st.multiselect("Sélectionnez un ou plusieurs fichiers :", fichiers)
        if fichiers_selectionnes:
            list_df = []
            for fichier in fichiers_selectionnes:
                df = lire_CIQ_csv(fichier_path=fichier, nom=fichier)
                if df is not None:
                    df = nettoyer_colonnes(df)
                    list_df.append(df)

            if list_df:
                CIQ = pd.concat(list_df, ignore_index=True)
                colonnes_dupliquees = CIQ.columns[CIQ.columns.duplicated()].tolist()
                if colonnes_dupliquees:
                    st.error(f"Doublons de colonnes détectés : {colonnes_dupliquees}")
                    st.stop()
                st.success(f"{len(list_df)} fichier(s) chargé(s), total : {CIQ.shape[0]} lignes.")
            else:
                st.warning("Aucun des fichiers sélectionnés n’a pu être chargé correctement.")
                st.stop()
        else:
            st.warning("Aucun fichier sélectionné.")
            st.stop()
    else:
        st.warning("Aucun fichier `lot*.csv` trouvé dans le répertoire courant.")
        st.stop()


st.dataframe(CIQ.head())

### Suppression des doublons éventuels sur Nickname/Date/Time/Sample No.

# Supprimer les doublons sur les colonnes spécifiées
CIQ_cleaned = CIQ.drop_duplicates(subset=["Nickname", "Date", "Time", "Sample No."])

st.write(f"Nombre de lignes initiales : {len(CIQ)}")
st.write(f"Nombre de lignes après suppression des doublons : {len(CIQ_cleaned)}")

CIQ=CIQ_cleaned

# st.dataframe(CIQ)

# === Chargement de la liste des champs ===
try:
    liste_champs_df = pd.read_csv("liste_champs.csv", sep=',', encoding="utf-8")
except UnicodeDecodeError:
    liste_champs_df = pd.read_csv("liste_champs.csv", sep=',', encoding="cp1252")

### renommer si unités différentes
liste_champs_df.rename(columns=lambda col: col.replace('(10^3/uL)', '(10^9/L)') if '(10^3/uL)' in col else col, inplace=True)
liste_champs_df.rename(columns=lambda col: col.replace('(10^6/uL)', '(10^12/L)') if '(10^6/uL)' in col else col, inplace=True)

if liste_champs_df.shape[1] == 1:
    colonnes_voulues = liste_champs_df.iloc[:, 0].dropna().astype(str).str.strip()
else:
    colonnes_voulues = liste_champs_df.columns.astype(str).str.strip()

# st.write("Colonnes du fichier colonnes_voulues :")
# st.write(CIQ.columns.tolist())

# Ajoute les colonnes manquantes à CIQ avec des valeurs NaN
for col in colonnes_voulues:
    if col not in CIQ.columns:
        CIQ[col] = np.nan  # ou pd.NA si tu préfères

# Réorganise les colonnes dans l’ordre de liste_champs
CIQ = CIQ[colonnes_voulues]

st.success(f"{len(colonnes_voulues)} colonnes définies dans CIQ (y compris les colonnes absentes ajoutées avec NaN).")


# st.write("Colonnes du fichier CIQ :")
# st.write(CIQ.columns.tolist())

# === Détection automatique des colonnes automate et lot ===
colonnes_automate = [col for col in CIQ.columns if 'nick' in col.lower()]
colonnes_lot = [col for col in CIQ.columns if 'sample' in col.lower() and 'no' in col.lower()]

if not colonnes_automate or not colonnes_lot:
    st.error("Colonnes 'automate' ou 'lot' non trouvées automatiquement.")
    st.write("Colonnes disponibles :", CIQ.columns.tolist())
    st.stop()

#col_automate = st.selectbox("Colonne automate :", colonnes_automate, key="automate")
col_automate = colonnes_automate[0]  # ou un autre index si tu veux la 2e, 3e colonne, etc.

# col_lot = st.selectbox("Colonne lot (sample no) :", colonnes_lot, key="lot")
col_lot = colonnes_lot[0]  # ou un autre index si tu veux la 2e, 3e colonne, etc.

CIQ[col_automate] = CIQ[col_automate].astype(str)


# === Création des colonnes lot_num et lot_niveau ===
CIQ['lot_num'] = CIQ[col_lot].astype(str).str[:18]
CIQ['lot_niveau'] = CIQ[col_lot].astype(str).str[18:22]
# Extraire Année
CIQ['Date'] = pd.to_datetime(CIQ['Date'], errors='coerce')
CIQ['Annee'] = CIQ['Date'].dt.year.astype("Int64")



st.subheader("Calcul des CV robustes, par paramètre, par analyseur, par année")

# === Choix du paramètre ===
choix_param = CIQ.columns[8:]  # adapter si besoin
param = st.selectbox("Choisissez le paramètre à étudier", choix_param)

# Sélection dynamique des critères

filt_automate = st.multiselect("Automate(s)", sorted(CIQ[col_automate].dropna().unique()), default=None)

# Forcer tout en chaînes pour uniformiser les types
niveaux_disponibles = sorted(CIQ['lot_niveau'].dropna().astype(str).unique())
# Définir les niveaux souhaités par défaut (aussi en str)
niveaux_defaut_souhaites = ['1101', '1102', '1103']
# Ne garder que les niveaux par défaut présents dans les options
niveaux_defaut_valides = [niveau for niveau in niveaux_defaut_souhaites if niveau in niveaux_disponibles]
# Affichage du multiselect sécurisé
filt_niveau = st.multiselect("Niveau(x) de lot", niveaux_disponibles, default=niveaux_defaut_valides)

lots_disponibles = sorted(CIQ['lot_num'].dropna().astype(str).unique())

filt_lot = st.multiselect("Numéro(s) de lot", lots_disponibles)

filt_annee = st.multiselect("Année(s)", sorted(CIQ['Annee'].dropna().unique()), default=None)


# Filtrage des données
data_filtrée = CIQ.copy()
if filt_automate:
    data_filtrée = data_filtrée[data_filtrée[col_automate].isin(filt_automate)]
if filt_niveau:
    data_filtrée = data_filtrée[data_filtrée['lot_niveau'].isin(filt_niveau)]
if filt_lot:
    data_filtrée = data_filtrée[data_filtrée['lot_num'].isin(filt_lot)]
if filt_annee:
    data_filtrée = data_filtrée[data_filtrée['Annee'].isin(filt_annee)]

# Conversion du paramètre sélectionné en float
data_filtrée[param] = pd.to_numeric(data_filtrée[param], errors='coerce')


# st.dataframe(data_filtrée)

# import fichier excel CV max sysmex / CV max recommandé


# Charger la première feuille en DataFrame
df_cv_max = pd.read_excel("CV_max_reco.xlsx", sheet_name=0, usecols=range(5))

# Afficher un aperçu du DataFrame
st.dataframe(df_cv_max)


# Agrégation par automate et niveau
grouped = data_filtrée.groupby([col_automate, 'lot_niveau','Annee'])[param].agg(
    n='count',
    Moyenne='mean',
    Mediane='median',
    Ecart_type='std',
    CV=cv,
    CV_IQR=cv_robuste_iqr,
    CV_IQR2=cv_robuste_iqr2,
    CV_MAD=cv_robuste_mad
).reset_index()

st.dataframe(grouped)

grouped['paramètre'] = param

# 1. Conversion forcée en string pour garantir la correspondance
grouped['lot_niveau'] = grouped['lot_niveau'].astype(str).str.strip()
df_cv_max['lot_niveau'] = df_cv_max['lot_niveau'].astype(str).str.strip()

# 2. On fait de même pour la colonne 'paramètre' par sécurité
grouped['paramètre'] = grouped['paramètre'].astype(str).str.strip()
df_cv_max['paramètre'] = df_cv_max['paramètre'].astype(str).str.strip()

# 3. Maintenant le merge fonctionnera
grouped = grouped.merge(
    df_cv_max[['paramètre', 'lot_niveau', 'CV_max_reco']], 
    on=['paramètre', 'lot_niveau'], 
    how='left'
)

st.subheader(f"Tableau des CV (CV classique / CV IQR / CV IQR robuste / CV MAD) pour {param}")
st.dataframe(grouped)


# Agrégation par automate, lot_num et niveau
grouped2 = data_filtrée.groupby([col_automate, 'lot_num','lot_niveau','Annee'])[param].agg(
    n='count',
    Moyenne='mean',
    Mediane='median',
    Ecart_type='std',
    CV=cv,
    CV_IQR=cv_robuste_iqr,
    CV_IQR2=cv_robuste_iqr2,
    CV_MAD=cv_robuste_mad
).reset_index()

grouped2['paramètre'] = param

# 1. Conversion forcée en string pour garantir la correspondance
grouped2['lot_niveau'] = grouped2['lot_niveau'].astype(str).str.strip()
df_cv_max['lot_niveau'] = df_cv_max['lot_niveau'].astype(str).str.strip()

# 2. On fait de même pour la colonne 'paramètre' par sécurité
grouped2['paramètre'] = grouped2['paramètre'].astype(str).str.strip()
df_cv_max['paramètre'] = df_cv_max['paramètre'].astype(str).str.strip()

# 3. Maintenant le merge fonctionnera
grouped2 = grouped2.merge(
    df_cv_max[['paramètre', 'lot_niveau', 'CV_max_reco']], 
    on=['paramètre', 'lot_niveau'], 
    how='left'
)

st.subheader(f"Tableau des CV (CV classique / CV IQR / CV IQR robuste / CV MAD) par Lot pour {param}")
st.dataframe(grouped2)

# Affichage des CV de tous les paramètres par analyseur et par niveau / avec filtre analyseur, lot_num, année
# st.dataframe(data_filtrée)

st.subheader("Tableau des CV (CV classique / CV IQR / CV IQR robuste / CV MAD) par analyseur et niveau de lot")

# Sélectionne les colonnes de l'index 8 à 125 pour permettre la conversion en numérique
# st.dataframe(CIQ.head())
colonnes_numeriques = CIQ.columns[8:125]

# Nettoyage et conversion en float
for col in colonnes_numeriques:
    CIQ[col] = (
        CIQ[col]
        .astype(str)
        .str.replace(',', '.', regex=False)
        .str.replace(r'[<>]', '', regex=True)
        .str.strip()
    )
    CIQ[col] = pd.to_numeric(CIQ[col], errors='coerce')

# Détection des colonnes numériques uniquement
params_all_numeriques = CIQ.select_dtypes(include=[np.number]).columns.tolist()

# Exclure les colonnes de comptage ou de type ID si nécessaire
params_all_numeriques = [col for col in params_all_numeriques if col not in ['n']]

# Liste par défaut
#params_all_visibles_par_défaut = [    
#    'WBC(10^9/L)','RBC(10^12/L)','HGB(g/L)','HCT(%)','MCV(fL)','MCH(pg)','MCHC(g/L)','PLT(10^9/L)','[RBC-O(10^12/L)]','[PLT-O(10^9/L)]','[PLT-F(10^9/L)]','IPF#(10^9/L)','[HGB-O(g/dL)]'
#    ]

# Sélecteur des paramètres à inclure
params_all_selectionnés = st.multiselect(
    "Paramètres à afficher",
    options=params_all_numeriques,
    default=params_all_numeriques
    #default=[p for p in params_all_numériques if p != "Annee"] # ✅ tous sauf 'Annee'
)

lots_uniques = data_filtrée["lot_num"].dropna().unique()
lots_str = ", ".join(map(str, sorted(lots_uniques)))

st.success(f"Liste des lots de CIQ inclus : {lots_str}")

# Mise en long format : chaque ligne = une mesure pour un paramètre donné
data_long = data_filtrée.melt(
    id_vars=['Nickname', 'lot_num', 'lot_niveau', 'Annee'],
    value_vars=params_all_selectionnés,
    var_name='paramètre',
    value_name='valeur'
)

# Conversion explicite en numérique
data_long["valeur"] = pd.to_numeric(data_long["valeur"], errors="coerce")

grouped3 = (
    data_long
    .groupby(['paramètre','Nickname','lot_niveau','Annee'])
    .apply(lambda g: pd.Series({
        "n": g["valeur"].count(),
        "Moyenne": g["valeur"].mean(),
        "Mediane": g["valeur"].median(),
        "Ecart_type": g["valeur"].std(),
        "CV": cv(g["valeur"].dropna()),
        "CV_IQR": cv_robuste_iqr(g["valeur"].dropna()),
        "CV_IQR2": cv_robuste_iqr2(g["valeur"].dropna()),
        "CV_MAD": cv_robuste_mad(g["valeur"].dropna())
    }))
    .reset_index()
)

# 1. Conversion forcée en string pour garantir la correspondance
grouped3['lot_niveau'] = grouped3['lot_niveau'].astype(str).str.strip()
df_cv_max['lot_niveau'] = df_cv_max['lot_niveau'].astype(str).str.strip()

# 2. On fait de même pour la colonne 'paramètre' par sécurité
grouped3['paramètre'] = grouped3['paramètre'].astype(str).str.strip()
df_cv_max['paramètre'] = df_cv_max['paramètre'].astype(str).str.strip()

# 3. Maintenant le merge fonctionnera
grouped3 = grouped3.merge(
    df_cv_max[['paramètre', 'lot_niveau', 'CV_max_reco']], 
    on=['paramètre', 'lot_niveau'], 
    how='left'
)


st.dataframe(grouped3)

# Graphs interactifs
#def plot_cv(y, title, ylabel):
#    fig = px.bar(grouped, x='lot_niveau', y=y, color=col_automate,
#                 barmode='group',
#                 hover_data=['n'],
#                 title=title,
#                 labels={y: ylabel, 'lot_niveau': 'Niveau de lot'})
#    st.plotly_chart(fig)

grouped['lot_annee'] = grouped['lot_niveau'].astype(str) + " (" + grouped['Annee'].astype(str) + ")"

def plot_cv(y, title, ylabel):
    fig = px.bar(grouped, x='lot_annee', y=y, color=col_automate,
                 barmode='group',
                 hover_data=['n', 'Annee', 'lot_niveau'],
                 title=title,
                 labels={y: ylabel, 'lot_annee': 'Niveau de lot (Année)'}
                )
    # Ajouter les lignes de seuil rouge
    # On ajoute une trace de type "scatter" (points reliés) pour le CV_max
    fig.add_scatter(
        x=grouped['lot_annee'], 
        y=grouped['CV_max_reco'], 
        name="CV Max recommandé",
        mode='markers', # 'markers' pour des points ou 'lines' si vous voulez relier
        marker=dict(color='red', symbol='line-ew', size=20, line_width=2),
        showlegend=True            
                )
    st.plotly_chart(fig)


st.subheader("Graphiques des CV")
plot_cv("CV", f"{param} : CV classique", "CV (%)")
plot_cv("CV_IQR", f"{param} : CV IQR", "CV (%)")
plot_cv("CV_IQR2", f"{param} : CV IQR robuste", "CV (%)")
plot_cv("CV_MAD", f"{param} : CV MAD", "CV (%)")

grouped2['lot_num2'] = grouped2['lot_num'].astype(str) + " (" + grouped2['lot_niveau'].astype(str) + ")"

def plot_cv2(y, title, ylabel):
    fig = px.bar(grouped2, x='lot_num2', y=y, color=col_automate,
                 barmode='group',
                 hover_data=['n', 'lot_num2','lot_niveau'],
                 title=title,
                 labels={y: ylabel, 'lot_num2':'Numéro de lot (Niveau)'}
                )

    # Ajouter les lignes de seuil rouge
    # On ajoute une trace de type "scatter" (points reliés) pour le CV_max
    fig.add_scatter(
        x=grouped2['lot_niveau'], 
        y=grouped2['CV_max_reco'], 
        name="CV Max recommandé",
        mode='markers', # 'markers' pour des points ou 'lines' si vous voulez relier
        marker=dict(color='red', symbol='line-ew', size=20, line_width=2),
        showlegend=True            
                )

    st.plotly_chart(fig)


st.subheader("Graphiques (2) des CV")
plot_cv2("CV", f"{param} : CV classique", "CV (%)")
plot_cv2("CV_IQR", f"{param} : CV IQR", "CV (%)")
plot_cv2("CV_IQR2", f"{param} : CV IQR robuste", "CV (%)")
plot_cv2("CV_MAD", f"{param} : CV MAD", "CV (%)")





# =======================
# ➕ Graphique Facets (CV_MAD par paramètre)
# =======================

st.subheader("CV (méthode MAD) par paramètre (moyenne de tous les CIQ)")


# Sélectionne les colonnes de l'index 8 à 125 pour permettre la conversion en numérique
# st.dataframe(CIQ.head())
colonnes_numeriques = CIQ.columns[8:125]

# Nettoyage et conversion en float
for col in colonnes_numeriques:
    CIQ[col] = (
        CIQ[col]
        .astype(str)
        .str.replace(',', '.', regex=False)
        .str.replace(r'[<>]', '', regex=True)
        .str.strip()
    )
    CIQ[col] = pd.to_numeric(CIQ[col], errors='coerce')
    
# st.write(CIQ.iloc[:, 8:125].dtypes)
# st.dataframe(CIQ.head())

# Détection des colonnes numériques uniquement
params_numeriques = CIQ.select_dtypes(include=[np.number]).columns.tolist()

# Exclure les colonnes de comptage ou de type ID si nécessaire
params_numeriques = [col for col in params_numeriques if col not in ['n']]

# Liste par défaut
params_visibles_par_défaut = [    
    'WBC(10^9/L)','RBC(10^12/L)','HGB(g/L)','HCT(%)','MCV(fL)','MCH(pg)','MCHC(g/L)','PLT(10^9/L)','[RBC-O(10^12/L)]','[PLT-O(10^9/L)]','[PLT-F(10^9/L)]','IPF#(10^9/L)','[HGB-O(g/dL)]'
    ]

# Sélecteur des paramètres à inclure dans les facets
params_selectionnés = st.multiselect(
    "Paramètres à afficher en facets",
    options=params_numeriques,
    default=[p for p in params_visibles_par_défaut if p in params_numeriques]
)

#st.write("Paramètres dans Excel :", df_cv_max['paramètre'].unique())
#st.write("Paramètres dans vos données :", params_selectionnés)

# Compilation des CV_MAD pour chaque paramètre
liste_dfs = []

for p in params_selectionnés:
    df_tmp = data_filtrée.groupby([col_automate, 'lot_niveau'])[p].agg(
        n='count',
        CV_MAD=cv_robuste_mad
    ).reset_index()
    
    # On ajoute le nom du paramètre pour pouvoir faire le merge
    df_tmp['paramètre'] = p
    df_tmp.rename(columns={'CV_MAD': 'CV'}, inplace=True)
    
    # Nettoyage des types avant le merge
    df_tmp['lot_niveau'] = df_tmp['lot_niveau'].astype(str).str.strip()
    df_tmp['paramètre'] = df_tmp['paramètre'].astype(str).str.strip()
    
    # Merge individuel pour récupérer le CV_max de ce paramètre
    df_tmp = df_tmp.merge(
        df_cv_max[['paramètre', 'lot_niveau', 'CV_max_reco']], 
        on=['paramètre', 'lot_niveau'], 
        how='left'
    )
    
    liste_dfs.append(df_tmp)



# Fusion de tous les DataFrames
df_facet = pd.concat(liste_dfs, ignore_index=True)

# On s'assure que le DataFrame est propre
df_facet['lot_niveau'] = df_facet['lot_niveau'].astype(str)

# 1. On définit le nombre de colonnes
n_cols = 3
n_params = len(params_selectionnés)
n_rows = math.ceil(n_params / n_cols)

# 2. Création du graphique en forçant l'ordre
fig_facet = px.bar(
    df_facet,
    x='lot_niveau',
    y='CV',
    color=col_automate,
    barmode='group',
    facet_col='paramètre',
    facet_col_wrap=n_cols,
    # Espacement horizontal (entre 0 et 1, par défaut très faible)
    facet_col_spacing=0.08, 
    # Espacement vertical (entre 0 et 1)
    facet_row_spacing=0.06,
    category_orders={
        "paramètre": params_selectionnés, # Ordre strict
        "lot_niveau": sorted(df_facet['lot_niveau'].unique().tolist())
    },
    title='CV MAD avec CV Max recommandés'
)

# 3. Ajout des seuils avec inversion de l'index des lignes
# Plotly Express place l'index 0 en BAS à GAUCHE.
for i, p in enumerate(params_selectionnés):
    sub_df = df_facet[df_facet['paramètre'] == p].drop_duplicates(subset=['lot_niveau'])
    sub_df = sub_df.dropna(subset=['CV_max_reco'])
    
    if not sub_df.empty:
        # CALCUL DE POSITION SPÉCIFIQUE PLOTLY
        # La colonne est simple :
        col_pos = (i % n_cols) + 1
        
        # La ligne doit être inversée car Plotly compte depuis le bas
        # Row 1 est en bas, Row N est en haut
        current_row_from_top = (i // n_cols)
        row_pos = n_rows - current_row_from_top
        
        fig_facet.add_trace(
            go.Scatter(
                x=sub_df['lot_niveau'],
                y=sub_df['CV_max_reco'],
                mode='markers',
                marker=dict(
                    symbol='line-ew', 
                    size=40, 
                    line=dict(width=3, color='red')
                ),
                name='CV max recommandés',
                legendgroup='Seuils',
                showlegend=(i == 0)
            ),
            row=int(row_pos), 
            col=int(col_pos)
        )
        

# 4. Ajustements
fig_facet.update_xaxes(
    showticklabels=True, 
    type='category', 
    title_text="Niveau" # Optionnel : ajoute un titre sous chaque axe
)
fig_facet.update_yaxes(matches=None, showticklabels=True)
fig_facet.update_layout(
    height=400 * n_rows, # On augmente un peu la hauteur par ligne
    margin=dict(t=100, b=100) # Plus de marge en bas pour les derniers labels
)

st.plotly_chart(fig_facet, width='stretch')

# Graphique facets avec Plotly Express
fig_facet2 = px.bar(
    df_facet,
    x='lot_niveau',
    y='CV',
    color=col_automate,
    barmode='group',
    facet_col='paramètre',
    facet_col_wrap=3,  # Nombre de colonnes dans la grille
    title='CV MAD par paramètre et par niveau de lot',
    facet_row_spacing=0.1,
    facet_col_spacing=0.1,
    height=1500,  # Plus grand pour laisser de la place
    labels={'CV': 'CV MAD (%)', 'lot_niveau': 'Niveau de lot'}
)

fig_facet2.update_yaxes(matches=None)  # axes Y indépendants


# Affiche tous les labels d’axe Y
for axis in fig_facet2.layout:
    if axis.startswith("yaxis"):
        fig_facet2.layout[axis].showticklabels = True
        fig_facet2.layout[axis].title = dict(text="CV MAD (%)")


# Forcer l’affichage de l’axe X et du titre sur chaque subplot
for axis_name in fig_facet2.layout:
    if axis_name.startswith("xaxis"):
        axis = fig_facet2.layout[axis_name]
        axis.showticklabels = True  # Affiche les ticks
        axis.title = dict(text="Niveau de lot")  # Titre de l'axe X


fig_facet2.update_layout(height=300 * ((len(params_selectionnés) - 1) // 3 + 1))  # ajuste la hauteur automatiquement
st.plotly_chart(fig_facet2)




# =======================
# ➕ Graphique Facets ( valeur paramètre) avec filtre année
# =======================

st.subheader("Distribution des valeurs de chaque paramètre")

if st.button("Afficher la distribution des valeurs des paramètres"):
    
    if len(params_selectionnés) == 0:
        st.warning("Veuillez sélectionner au moins un paramètre.")
    else:
        df_facet = data_filtrée[[col_automate, 'lot_niveau','Annee'] + params_selectionnés].copy()
        df_facet['lot_niveau'] = df_facet['lot_niveau'].astype(str)
        
    
        df_melted = df_facet.melt(
            id_vars=[col_automate, 'lot_niveau','Annee'],
            value_vars=params_selectionnés,
            var_name='paramètre',
            value_name='valeur'
        )
    
        df_melted[col_automate] = df_melted[col_automate].astype(str)
        df_melted['lot_niveau'] = df_melted['lot_niveau'].astype(str)
        df_melted = df_melted.dropna()
    
        
    
        fig_facet = px.box(
            df_melted,
            x='lot_niveau',
            y='valeur',
            color=col_automate,
            facet_col='paramètre',
            facet_col_wrap=3,
            title="Distribution des paramètres par niveau de lot",
            facet_row_spacing=0.1,
            facet_col_spacing=0.1,
            height=1500,
            labels={
                'valeur': 'Valeur mesurée',
                'lot_niveau': 'Niveau de lot'
            }
        )
        fig_facet.update_yaxes(matches=None)
    
        for axis in fig_facet.layout:
            if axis.startswith("yaxis"):
                fig_facet.layout[axis].showticklabels = True
                fig_facet.layout[axis].title = dict(text="Valeur")
    
        for axis_name in fig_facet.layout:
            if axis_name.startswith("xaxis"):
                axis = fig_facet.layout[axis_name]
                axis.showticklabels = True
                axis.title = dict(text="Niveau de lot")
    
        fig_facet.update_layout(height=300 * ((len(params_selectionnés) - 1) // 3 + 1))  # ajuste la hauteur automatiquement
        st.plotly_chart(fig_facet)


### ---------------- #####
### EEQ => IM ###

st.title("Incertitudes élargies")

# === Fonction générique de lecture EEQ (sans ignorer la première ligne) ===
def lire_fichier_eeq(fichier_path=None, contenu_brut=None, nom=""):
    """
    Lit un fichier EEQ en gardant la première ligne (en-tête),
    en utilisant le séparateur ';' et encodage ISO-8859-1.
    """
    try:
        if fichier_path:
            with open(fichier_path, 'r', encoding='ISO-8859-1', errors='replace') as f:
                content = f.read().splitlines()
        elif contenu_brut:
            content = contenu_brut.decode('ISO-8859-1', errors='replace').splitlines()
        else:
            return None

        if len(content) < 1:
            st.warning(f"Le fichier {nom} semble vide ou mal formaté.")
            return None

        content_str = StringIO('\n'.join(content))  # On garde toutes les lignes
        df = pd.read_csv(content_str, sep=';', on_bad_lines='skip')
        return df

    except Exception as e:
        st.error(f"Erreur lecture du fichier {nom or fichier_path} : {e}")
        return None

# === Choix de la source de données EEQ ===
choix_eeq = st.radio("Source du fichier EEQ :", ["Importer un fichier EEQ", "Utiliser un fichier EEQ par défaut"])

if choix_eeq == "Importer un fichier EEQ":
    uploaded_eeq = st.file_uploader("Importer fichier EEQ (exportEEQ1952.csv)", type=["csv"])
    
    if uploaded_eeq:
        EEQ = lire_fichier_eeq(contenu_brut=uploaded_eeq.read(), nom=uploaded_eeq.name)
        if EEQ is not None:
            st.success(f"Fichier EEQ importé avec succès : {uploaded_eeq.name}")
            # st.dataframe(EEQ.head())
        else:
            st.stop()
    else:
        st.stop()

elif choix_eeq == "Utiliser un fichier EEQ par défaut":
    EEQ = lire_fichier_eeq(fichier_path="exportEEQ1952.csv")
    if EEQ is not None:
        st.success("Fichier EEQ par défaut chargé depuis `exportEEQ1952.csv`.")
        # st.dataframe(EEQ.head())
    else:
        st.stop()
   
    # === Traitement données EEQ ===
    # Exemple: filtrer et ajouter 'Nickname', 'variable', 'Date' (adapter selon colonnes EEQ)
    # On suppose EEQ a colonnes 'Date', 'App', 'Anonymat', 'Analyte', 'Biais |c| pairs', etc.
    
    # Convertir date EEQ en datetime
    if 'Date' in EEQ.columns:
        EEQ['Date'] = pd.to_datetime(EEQ['Date'], dayfirst=True, errors='coerce')
    else:
        st.error("Colonne 'Date' introuvable dans EEQ.")
        st.stop()
    
    # Filtrer app NK9 (exemple)
    EEQ = EEQ[EEQ['App'] == 'NK9']
    
    # Ajouter Nickname en fonction date et Anonymat (adaptation directe de R -> Python)
    def assign_nickname(row):
        d = row['Date']
        a = row['Anonymat']
        if pd.isna(d) or pd.isna(a):
            return np.nan
        if d >= pd.Timestamp("2013-01-01") and d <= pd.Timestamp("2023-05-10"):
            if a == "1952": return "ATHOS-1"
            elif a == "1952A": return "PORTHOS-2"
            elif a == "1952B": return "ARAMIS-3"
        elif d >= pd.Timestamp("2023-05-11") and d <= pd.Timestamp("2023-12-10"):
            if a == "1952": return "XN-9100-1-A"
            elif a == "1952A": return "XN-9100-2-A"
            elif a == "1952B": return "XN-9100-3-A"
        elif d >= pd.Timestamp("2023-12-11"):
            if a == "1952": return "XR-ISIS-A"
            elif a == "1952A": return "XR-OSIRIS-A"
            elif a == "1952B": return "XR-ANUBIS-A"
            elif a == "1952C": return "XN-1000-1-A"
        return np.nan

    EEQ['Nickname'] = EEQ.apply(assign_nickname, axis=1)

    # Ajouter variable selon Analyte (mapping R -> Python)
    analyte_map = {
        "Hématies (optique)": "[RBC-O(10^12/L)]",
        "Hématies (impédance)": "RBC(10^12/L)",
        "Hématocrite": "HCT(%)",
        "Hémoglobine": "HGB(g/L)",
        "IDR": "RDW-CV(%)",
        "VGM": "MCV(fL)",
        "CCMH": "MCHC(g/dL)",
        "TGMH": "MCH(pg)",
        "Plaquettes (fluorescence)": "[PLT-F(10^9/L)]",
        "Plaquettes (impédance)": "PLT(10^9/L)",
        "Plaquettes (optique)": "[PLT-O(10^9/L)]",
        "VPM": "MPV(fL)",
        "Frac Plaq immatures (IPF)": "IPF(%)",
        "Réticulocytes": "RET%(%)",
        "Réticulocytes (abs)": "RET#(10^9/L)",
        "Teneur Hb Réti (Ret-He)": "RET-He(pg)",
        "Frac Réti immatures (IRF)": "IRF(%)",
        "TGMH optique (GR-He)": "RBC-He(pg)",
        "R-MFV (Volume Erythro. le plus fréquent)": "R-MFV(fL)",
        "Leucocytes": "WBC(10^9/L)",
        "Poly. Neutrophiles": "NEUT%(%)",
        "Poly. Neutrophiles (abs)": "NEUT#(10^9/L)",
        "Poly. Eosinophiles": "EO%(%)",
        "Poly. Eosinophiles (abs)": "EO#(10^9/L)",
        "Poly. Basophiles": "BASO%(%)",
        "Poly. Basophiles (abs)": "BASO#(10^9/L)",
        "Lymphocytes": "LYMPH%(%)",
        "Lymphocytes (abs)": "LYMPH#(10^9/L)",
        "Monocytes": "MONO%(%)",
        "Monocytes (abs)": "MONO#(10^9/L)"       
        }
    EEQ['variable'] = EEQ['Analyte'].map(analyte_map)

    # Extraire Année
    EEQ['Annee'] = EEQ['Date'].dt.year
 
    # Joindre CIQ et EEQ pour la même variable, Nickname (automate), année
    # Dans CIQ, on doit avoir colonne Année à créer (par exemple date d’analyse)
    # Ici on suppose CIQ a une colonne date, sinon on crée Année manuellement (à adapter)
    if 'Date' in CIQ.columns:
        CIQ['Date'] = pd.to_datetime(CIQ['Date'], errors='coerce')
        CIQ['Annee'] = CIQ['Date'].dt.year
    else:
        st.warning("Pas de colonne Date dans CIQ : Année non disponible.")
        CIQ['Annee'] = 0  # placeholder
    
    # Calcul du biais moyen absolu par groupe
    # st.dataframe(EEQ.head())

    EEQ = EEQ.rename(columns={"variable": "Paramètre"})

    if "Paramètre" in EEQ.columns:
        EEQ["Biais |c| pairs"] = (
        EEQ["Biais |c| pairs"]
        .str.replace(",", ".", regex=False)  # remplacer la virgule par un point
        .astype(float)                       # convertir en float
)
    st.dataframe(EEQ.head())
 



    # st.dataframe(CIQ)

    colonnes_valeurs = CIQ.columns[8:125]  
    
    CIQ_long = CIQ.melt(
        id_vars=["Nickname", "lot_niveau", "Annee"],
        value_vars=colonnes_valeurs,
        var_name="Paramètre",
        value_name="Valeur"
    )
    # st.dataframe(CIQ_long.head())
    
    CIQ_moyennes = (
    CIQ_long.groupby(["Nickname", "Paramètre", "lot_niveau", "Annee"])
    .agg(moy_valeur=("Valeur", lambda x: pd.to_numeric(x, errors="coerce").mean()))
    .reset_index()
)

    # st.dataframe(CIQ_moyennes.head())
    
    EEQ["Resultat"] = (
        EEQ["Resultat"]
        .astype(str)  # au cas où il y aurait des nombres mélangés avec des strings
        .str.replace(",", ".", regex=False)
    )

    EEQ["Resultat"] = pd.to_numeric(EEQ["Resultat"], errors="coerce")
    CIQ_moyennes["moy_valeur"] = pd.to_numeric(CIQ_moyennes["moy_valeur"], errors="coerce")

    # Application sur ton DataFrame EEQ
    EEQ["lot_niveau_proche"] = EEQ.apply(lambda row: trouver_lot_niveau_proche(row, CIQ_moyennes), axis=1)
   
    # st.dataframe(EEQ)
   
    biais_moyen = (
    EEQ.groupby(["Nickname", "Paramètre", "Annee","lot_niveau_proche"])
    .agg(
    moy_biais=("Biais |c| pairs", lambda x: np.mean(np.abs(x.dropna()))),
    sd_biais=("Biais |c| pairs", lambda x: np.std(x.dropna()))
    ) 
    .reset_index()
    )
    
   
    # st.write("Aperçu des colonnes  :", biais_moyen.columns.tolist())
    # st.dataframe(biais_moyen.head())
    
    
    CIQ_grouped = CIQ_long.groupby(["Nickname", "lot_niveau", "Annee", "Paramètre"]).agg(
        Moyenne=('Valeur', 'mean'),
        Médiane=('Valeur', 'median'),
        Ecart_type=('Valeur', 'std'),
        N=('Valeur', 'count'),
        CV_classique=('Valeur',cv),
        SD_classique=('Valeur',sd),
        CV_IQR=('Valeur',cv_robuste_iqr),
        SD_IQR=('Valeur',sd_robuste_iqr),
        CV_IQR2=('Valeur',cv_robuste_iqr2),
        SD_IQR2=('Valeur',sd_robuste_iqr2),
        CV_MAD=('Valeur',cv_robuste_mad),
        SD_MAD=('Valeur',mad)
        ).reset_index()

    # st.dataframe(CIQ_grouped)
    
    # CIQ_grouped_1102 = CIQ_grouped[CIQ_grouped["lot_niveau"] == "1102"]
    # st.dataframe(CIQ_grouped_1102)
    
    # df_IM = pd.merge(
    # biais_moyen,
    # CIQ_grouped_1102,
    # on=["Nickname", "Paramètre", "Annee"],
    # how="inner"  # ou "left", "right", "outer" selon ton besoin
# )
    limites_en_pourcentage = {
    "WBC(10^9/L)": 15.49,
    "RBC(10^12/L)": 4.4,
    "HGB(g/L)": 4.19,
    "HCT(%)": 3.97,
    "PLT(10^9/L)": 13.4,
    "[PLT-F(10^9/L)]": 13.4,
    "RET#(10^9/L)": 16.8,
    "MCV(fL)": 2.42,
    "LYMPH#(10^9/L)": 17.6,
    "MONO#(10^9/L)": 27.9,
    "BASO#(10^9/L)": 38.5,
    "EO#(10^9/L)": 37.1,
    "NEUT#(10^9/L)": 23.35
    }
    
    df_IM = pd.merge(
        biais_moyen,
        CIQ_grouped,
        left_on=["Nickname", "Paramètre", "Annee", "lot_niveau_proche"],
        right_on=["Nickname", "Paramètre", "Annee", "lot_niveau"],
        how="inner"  # ou "left", "right", "outer" selon ton besoin
    )

    df_IM = df_IM.drop(columns="lot_niveau")

    # Crée une colonne de limite absolue si une limite en % est connue
    def calculer_limite_absolue(row):
        param = row['Paramètre']
        moyenne = row['Moyenne']
        if param in limites_en_pourcentage and pd.notnull(moyenne):
            return moyenne * limites_en_pourcentage[param] / 100
        else:
            return np.nan

    df_IM['limite_accept'] = df_IM.apply(calculer_limite_absolue, axis=1)

    
    # st.dataframe(df_IM)
    
# Calcul incertitudes
    # Colonnes proposées
    options_sd = ['SD_classique', 'SD_IQR', 'SD_IQR2', 'SD_MAD']

    # Sélecteur unique
    choix_sd = st.selectbox("Choisissez le type de CV pour calculer u_CIQ :", options_sd)

    # Affecter la colonne choisie à u_CIQ si elle existe dans df_IM
    if choix_sd in df_IM.columns:
        df_IM['u_CIQ'] = df_IM[choix_sd]
        st.write(f"Colonne `{choix_sd}` utilisée pour calculer 'u_CIQ'")
        # st.dataframe(df_IM[['u_CIQ']].head())
    else:
        st.warning(f"La colonne {choix_sd} n'existe pas dans les données.")
    
    
    df_IM['u_biais'] = df_IM['sd_biais']
    df_IM['u_total'] = np.sqrt(df_IM['u_biais']**2 + df_IM['u_CIQ']**2)
    df_IM['U'] = df_IM['u_total'] * 2  # élargie (k=2)
    df_IM['U%'] = 100 * df_IM['U'] / df_IM['Moyenne']

    st.dataframe(df_IM)
    


    # =======================
    # ➕ Graphique Facets (IM)
    # =======================

    st.subheader("Facets : Incertitudes de Mesure")

    # Liste par défaut
    params_visibles_par_défaut = [    
        'WBC(10^3/uL)','RBC(10^6/uL)','HGB(g/L)','HCT(%)','MCV(fL)','MCH(pg)','MCHC(g/L)','PLT(10^3/uL)','[RBC-O(10^6/uL)]','[PLT-O(10^3/uL)]','[PLT-F(10^3/uL)]','IPF#(10^3/uL)','[HGB-O(g/dL)]'
        ]

    # Choix des paramètres via menu déroulant
    # parametres_disponibles = df_IM['Paramètre'].unique()
    # param_selectionnes = st.multiselect("Sélectionnez un ou plusieurs paramètres",options=parametres_disponibles,
    #    default=[p for p in params_visibles_par_défaut if p in parametres_disponibles])

     # Choix des paramètres via menu déroulant
    parametres_disponibles = df_IM['Paramètre'].unique()
    param_selectionnes = st.multiselect("Sélectionnez un ou plusieurs paramètres",options=parametres_disponibles,
        default=None)

    # Filtrer selon les paramètres choisis
    df_IM_filtré = df_IM[df_IM['Paramètre'].isin(param_selectionnes)]

    # Graphique faceté
    st.subheader("Incertitudes élargies (U) par année et par analyseur")

    facet_row_order = sorted(df_IM_filtré["lot_niveau_proche"].unique(), key=lambda x: str(x))
    facet_col_order = sorted(df_IM_filtré["Paramètre"].unique(), key=lambda x: str(x))

    # st.write("Données uniques pour lot_niveau_proche et Paramètre:", df_IM_filtré[["lot_niveau_proche", "Paramètre"]].drop_duplicates())

    
    fig_IM = px.bar(
        df_IM_filtré,
        x="Annee",
        y="U",
        color="Nickname",
        facet_row="lot_niveau_proche",     # ➜ 1 ligne par lot_niveau_proche
        facet_col="Paramètre",             # ➜ 1 colonne par paramètre
        facet_row_spacing=0.1,
        facet_col_spacing=0.15,
        barmode="group",
        title="Incertitude élargie par Paramètre et par Année",
        labels={"U": "Incertitude élargie", "Annee": "Année"},
        category_orders={"lot_niveau_proche": facet_row_order}  # 🔥 Assure l'ordre correct des lots
    )

    # Axe y indépendant pour chaque facette
    fig_IM.update_yaxes(matches=None)

    for axis in fig_IM.layout:
        if axis.startswith("yaxis"):
            fig_IM.layout[axis].showticklabels = True
            fig_IM.layout[axis].title = dict(text="Incertitude Elargie (U)")

    for axis_name in fig_IM.layout:
        if axis_name.startswith("xaxis"):
            axis = fig_IM.layout[axis_name]
            axis.showticklabels = True
            axis.title = dict(text="Annee")

    nb_lots = df_IM_filtré["lot_niveau_proche"].nunique()
    nb_params = df_IM_filtré["Paramètre"].nunique()

    #fig_IM.update_layout(
    #   height=max(300, 250 * len(facet_row_order)),  # Ajuste en fonction du nombre réel de lignes
    #    showlegend=True,
    #    facet_row_wrap=1  # 🔥 Forcer une seule série de facettes
    #)

  
    fig_IM.update_layout(
        height=max(300, 250 * len(facet_row_order)),  # Ajuste en fonction du nombre réel de lignes
        showlegend=True,  # 🔥 Forcer une seule série de facettes
    #    height=max(300, 250 * nb_lots * nb_params),  # Ajustement plus fin si tu veux
    )

import plotly.graph_objects as go
# 📌 Récupérer l'ordre des facettes **tel que Plotly Express les affiche**


# Inversion de l'indexation des facettes pour correspondre au vrai mapping
# facet_row_order_reversed = list(reversed(facet_row_order))  # 🔄 Inverse l'ordre des lignes
# st.write("Ordre des facettes  - lot_niveau_proche:", facet_row_order)
# st.write("Ordre des facettes inversé - lot_niveau_proche:", facet_row_order_reversed)


# Ajout des points pour 'limite_accept' en respectant l'affichage des facettes
#for lot in facet_row_order:
 #   for param in facet_col_order:
  #      df_subset = df_IM_filtré[(df_IM_filtré["lot_niveau_proche"] == lot) & (df_IM_filtré["Paramètre"] == param)]
   #     
    #    fig_IM.add_trace(
     #       go.Scatter(
      #          x=df_subset["Annee"],
       #         y=df_subset["limite_accept"],
        #        mode="markers",
         #       marker=dict(color="red", size=8),
          #      name=f"Limite acceptée - {lot}, {param}",
           # ),
            #row=facet_row_order_reversed.index(lot) + 1,  # 🔥 Utilisation correcte des indices réels
            #col=facet_col_order.index(param) + 1  # 🔥 Alignement parfait des colonnes
        #)

# Palette de couleurs (tu peux choisir d'autres couleurs ou symboles)
# colors = px.colors.qualitative.Plotly
# symbols = ["circle", "square", "diamond", "star", "triangle-up", "cross"]

nicknames = df_IM_filtré["Nickname"].unique()

# for lot in facet_row_order:
#    for param in facet_col_order:
#       for i, nickname in enumerate(nicknames):
#           df_subset = df_IM_filtré[
#               (df_IM_filtré["lot_niveau_proche"] == lot) &
#               (df_IM_filtré["Paramètre"] == param) &
#               (df_IM_filtré["Nickname"] == nickname)
#           ]
#           if df_subset.empty:
#               continue

#           fig_IM.add_trace(
#               go.Scatter(
#                   x=df_subset["Annee"],
#                   y=df_subset["limite_accept"],
#                   mode="markers",
#                   marker=dict(
#                       color=colors[i % len(colors)],
#                       size=10,
#                       symbol=symbols[i % len(symbols)],
#                       line=dict(width=1, color="black")  # contour noir pour bien voir
#                   ),
#                   name=f"Limite acceptée - {nickname}",
#                   showlegend=True,
#                   hovertemplate=(
#                       f"Limite: %{{y}}<br>"
#                       f"Année: %{{x}}<br>"
#                       f"Analyseur: {nickname}<extra></extra>"
#                   )
#               ),
#               row=facet_row_order_reversed.index(lot) + 1,
#               col=facet_col_order.index(param) + 1
#           )

    # fig_IM.update_layout(height=300 * len(param_selectionnes))
st.plotly_chart(fig_IM, use_container_width=True)


### Méthode 2 pour graph ###

    # Charger le fichier
#df = pd.read_csv("2025-06-11T06-52_export.csv")

# Garder uniquement les colonnes utiles et supprimer les lignes avec U ou Annee manquants
df_plot = df_IM_filtré[['Annee', 'Nickname', 'lot_niveau_proche', 'U', 'limite_accept']].dropna(subset=['U', 'Annee'])

# Transformer en format long pour U et limite_accept
df_long = df_plot.melt(
    id_vars=['Annee', 'Nickname', 'lot_niveau_proche'],
    value_vars=['U', 'limite_accept'],
    var_name='Type',
    value_name='Valeur'
)

df_long = df_long.rename(columns={"Nickname": "Analyseur"})
df_long['Type'] = df_long['Type'].replace({
    'U': 'U',
    'limite_accept': 'Limites acceptables'
})

# st.dataframe(df_long)

color_discrete_map = {
    'U': 'royalblue',
    'Limites acceptables': 'red'
}

pattern_shape_map = {
    'U': '',
    'Limites acceptables': '/'
}


titre_graph = f"Incertitudes de mesure (U) pour {param_selectionnes} et limites acceptables par année, par analyseur et par niveau de lot"

# Sélection interactive des analyseurs
analyseurs_disponibles = df_long['Analyseur'].unique()
analyseurs_selectionnes = st.multiselect(
    "Sélectionnez les analyseurs à afficher :",
    options=sorted(analyseurs_disponibles),
    default=sorted(analyseurs_disponibles)
)
df_long_filtre = df_long[df_long['Analyseur'].isin(analyseurs_selectionnes)]

# 🎨 Autres motifs disponibles : '' (plein) '/' (diagonal 45°) '\\' (diagonal -45°) 'x' (croix) '-' (horizontal) '|' (vertical) '+' (croix pleine) '.' (points)

# Création du graphique en barres interactif
fig_IM2 = px.bar(
    df_long_filtre,
    x='Analyseur',
    facet_row='lot_niveau_proche',
    y='Valeur',
    color='Type',
    barmode='group',
    facet_col='Annee',
    facet_col_wrap=3,
    pattern_shape='Type',
    pattern_shape_map={
        'U': '',
        'limite_accept': '/'
    },
    color_discrete_map={
        'U': 'royalblue',
        'limite_accept': 'red'
    },
    title=titre_graph,
    labels={'Annee': 'Année', 'Valeur': 'Valeur', 'Type': 'Type de mesure'},
    hover_data=['Analyseur']
)

fig_IM2.update_layout(height=600, legend_title_text='Type')
#fig_IM2.update_traces(mode="lines+markers")
st.plotly_chart(fig_IM2, use_container_width=True)
