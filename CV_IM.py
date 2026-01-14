import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from io import StringIO
# from datetime import datetime
import glob
import os
import plotly.graph_objects as go
import math

st.set_page_config(
    layout="wide"
)

st.sidebar.header("Data source")

tab_CV_intralot, tab_CV_interlot, tab_CVref, tab_IM , tab_data= st.tabs(
    ["📊 Tableaux/Graphiques CV intra-lot", "📊 Tableaux/Graphiques CV inter-lot", "📈 Calcul CV robuste (CV référence)", "📈 Calcul des incertitudes de mesures (IM)","Data source"]
)


# === Def import CIQ csv ===

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

def sd_pooled(df, param_name):
    """
    Calcule l'écart-type combiné (Pooled SD) pour un paramètre,
    en groupant par numéro de lot pour neutraliser les bais inter-lots.
    """
    # On s'assure que les données sont numériques
    df[param_name] = pd.to_numeric(df[param_name], errors='coerce')
    
    # On groupe par 'lot_num' pour avoir le SD et le n de chaque lot
    stats = df.groupby('lot_num')[param_name].agg(['std', 'count']).dropna()
    
    if stats.empty or len(stats) < 1:
        return np.nan
    
    # Calcul du numérateur : Somme des (n-1) * SD^2
    numerator = sum((stats['count'] - 1) * (stats['std']**2))
    
    # Calcul du dénominateur : Somme des n - nombre de lots (k)
    denominator = sum(stats['count']) - len(stats)
    
    if denominator <= 0:
        return np.nan
        
    return np.sqrt(numerator / denominator)

def calcul_sd_pooled_robuste(group):
    """
    Calcule le SD Pooled Robuste (basé sur la MAD)
    """
    if 'lot_num' not in group.columns or group.empty:
        return np.nan

    # SD robuste par lot (MAD * 1.4826)
    def get_mad_sd(x):
        x = pd.to_numeric(x, errors='coerce').dropna()
        if len(x) < 2: 
            return np.nan
        return np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826

    # Calcul du SD robuste et effectif par lot
    stats = group.groupby('lot_num')['Valeur'].agg([get_mad_sd, 'count']).dropna()
    stats.columns = ['sd_rob', 'n']
    stats = stats[stats['n'] > 1] # On ignore les lots isolés
    
    if stats.empty:
        return np.nan
        
    # Formule du Pooled Variance -> Pooled SD
    numerator = ((stats['n'] - 1) * (stats['sd_rob']**2)).sum()
    denominator = stats['n'].sum() - len(stats)
    
    if denominator <= 0:
        return np.nan
        
    return np.sqrt(numerator / denominator)

def calcul_sd_pooled_custom(group):
    """Fonction corrigée pour le calcul du SD Pooled"""
    # On vérifie si 'lot_num' est présent dans les colonnes du groupe
    # Si include_groups=False a été utilisé, il faut s'assurer que lot_num n'était pas dans les clés
    target_col = 'lot_num'
    
    if target_col not in group.columns:
        return np.nan

    # Calcul des stats par lot
    stats = group.groupby(target_col)['Valeur'].agg(['std', 'count']).dropna()
    
    if stats.empty:
        return np.nan
    
    # On ne garde que les lots ayant au moins 2 mesures pour avoir un SD calculable
    stats = stats[stats['count'] > 1]
    
    if stats.empty:
        return np.nan
        
    numerator = ((stats['count'] - 1) * (stats['std']**2)).sum()
    denominator = stats['count'].sum() - len(stats)
    
    return np.sqrt(numerator / denominator) if denominator > 0 else np.nan

def calculate_cv_pooled_robust_internal(df, param_col, lot_col):
    """Calcule le CV Pooled Robuste pour un sous-groupe de données"""
    def get_mad_sd(x):
        x = pd.to_numeric(x, errors='coerce').dropna()
        if len(x) < 2: 
            return np.nan
        return np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826

    # Groupement par lot réel pour le calcul pooled
    stats = df.groupby(lot_col)[param_col].agg([get_mad_sd, 'count']).dropna()
    stats.columns = ['sd_rob', 'n']
    stats = stats[stats['n'] > 1]
    
    if stats.empty: 
        return np.nan
    
    numerator = ((stats['n'] - 1) * (stats['sd_rob']**2)).sum()
    denominator = stats['n'].sum() - len(stats)
    
    if denominator <= 0: 
        return np.nan
    
    sd_pooled_rob = np.sqrt(numerator / denominator)
    moyenne_globale = df[param_col].mean()
    
    return (sd_pooled_rob / moyenne_globale) * 100 if moyenne_globale != 0 else np.nan

# === Fonctions de calcul des LT-CV

def cv_long_terme_mad(x):
    """
    Calcule le CV Long Terme (LT-CV) de manière robuste.
    Prend l'ensemble des points d'un niveau (tous lots confondus).
    """
    x = pd.to_numeric(x, errors='coerce').dropna()
    if len(x) < 2:
        return np.nan
    
    med_globale = np.nanmedian(x)
    if med_globale == 0:
        return np.nan
    
    # On calcule la MAD sur la totalité des données du niveau
    mad_totale = np.nanmedian(np.abs(x - med_globale)) * 1.4826
    
    return (mad_totale / med_globale) * 100


# def Trouver le niveau de lot le plus proche

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

# === def Graphiques CV intra-lot ===

def plot_cv(y, title, ylabel):
    fig = px.bar(grouped, x='lot_niveau', y=y, color=col_automate,
                 barmode='group',
                 hover_data=['n', 'Annee', 'lot_niveau'],
                 title=title,
                 labels={y: ylabel, 'lot_niveau': 'Niveau de lot'}
                )
    # Ajouter les lignes de seuil rouge
    # On ajoute une trace de type "scatter" (points reliés) pour le CV_max
    fig.add_scatter(
        x=grouped['lot_niveau'], 
        y=grouped['CV_max_reco'], 
        name="CV Max recommandé",
        mode='markers', # 'markers' pour des points ou 'lines' si vous voulez relier
        marker=dict(color='red', symbol='line-ew', size=20, line_width=2),
        showlegend=True            
                )
    ordre_niveaux = sorted(grouped['lot_niveau'].unique())

    fig.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=ordre_niveaux
    )
    st.plotly_chart(fig)

def plot_cvinter(y, title, ylabel):
    fig = px.bar(grouped_cvinter, x='lot_num2', y=y, color=col_automate,
                barmode='group',
                hover_data=['n', 'lot_num2','lot_niveau'],
                title=title,
                labels={y: ylabel, 'lot_num2':'Numéro de lot (Niveau)'}
                )

    # Ajouter les lignes de seuil rouge
    # On ajoute une trace de type "scatter" (points reliés) pour le CV_max
    fig.add_scatter(
        x=grouped_cvinter['lot_niveau'], 
        y=grouped_cvinter['CV_max_reco'], 
        name="CV Max recommandé",
        mode='markers', # 'markers' pour des points ou 'lines' si vous voulez relier
        marker=dict(color='red', symbol='line-ew', size=20, line_width=2),
        showlegend=True            
                )

    st.plotly_chart(fig)


# === Choix de la source de données ===
choix_source = st.sidebar.radio(
    "Choisissez la source des données de CIQ (csv au format sysmex):",
    ["Importer des fichiers CSV", "Utiliser les données par défaut", "Rechercher un fichier lot*.csv localement"]
)

if choix_source == "Importer des fichiers CSV":
    uploaded_files = st.sidebar.file_uploader("Importer un ou plusieurs fichiers CSV", type=["csv"], accept_multiple_files=True)

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
                st.sidebar.error(f"Doublons de colonnes détectés : {colonnes_dupliquees}")
                st.stop()
            st.sidebar.success(f"{len(list_df)} fichier(s) chargé(s), total : {CIQ.shape[0]} lignes.")
        else:
            st.sidebar.warning("Aucun fichier n'a pu être chargé correctement.")
            st.stop()
    else:
        st.stop()

elif choix_source == "Utiliser les données par défaut":
    df = lire_CIQ_csv(fichier_path="lot_default.csv")
    if df is not None:
        df = nettoyer_colonnes(df)
        colonnes_dupliquees = df.columns[df.columns.duplicated()].tolist()
        if colonnes_dupliquees:
            st.sidebar.error(f"Doublons de colonnes détectés : {colonnes_dupliquees}")
            st.stop()
        CIQ = df
        st.sidebar.success("Données par défaut chargées depuis `lot_default.csv`.")
    else:
        st.sidebar.warning("Impossible de charger `lot_default.csv`.")
        st.stop()

elif choix_source == "Rechercher un fichier lot*.csv localement":
    fichiers = glob.glob("lot*.csv")
    if fichiers:
        fichiers_selectionnes = st.sidebar.multiselect("Sélectionnez un ou plusieurs fichiers :", fichiers)
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
                    st.sidebar.error(f"Doublons de colonnes détectés : {colonnes_dupliquees}")
                    st.stop()
                st.sidebar.success(f"{len(list_df)} fichier(s) chargé(s), total : {CIQ.shape[0]} lignes.")
            else:
                st.sidebar.warning("Aucun des fichiers sélectionnés n’a pu être chargé correctement.")
                st.stop()
        else:
            st.sidebar.warning("Aucun fichier sélectionné.")
            st.stop()
    else:
        st.sidebar.warning("Aucun fichier `lot*.csv` trouvé dans le répertoire courant.")
        st.stop()


# st.dataframe(CIQ.head())

### Suppression des doublons éventuels sur Nickname/Date/Time/Sample No.

# Supprimer les doublons sur les colonnes spécifiées
CIQ_cleaned = CIQ.drop_duplicates(subset=["Nickname", "Date", "Time", "Sample No."])

st.sidebar.write(f"Nombre de lignes initiales : {len(CIQ)}")
st.sidebar.write(f"Nombre de lignes après suppression des doublons : {len(CIQ_cleaned)}")

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

#  st.success(f"{len(colonnes_voulues)} colonnes définies dans CIQ.")


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


# === import fichier excel CV max sysmex / CV max recommandé ===
# Charger la première feuille en DataFrame
df_cv_max = pd.read_excel("CV_max_reco.xlsx", sheet_name=0, usecols=range(5))

with tab_data:
    st.title("Data")
    with st.expander("Data brutes (CIQ Sysmex)"):
        st.dataframe(CIQ)

   # == Choix du numéro de lot ===

    st.subheader("Lots présents dans le jeu de données")

    # Tableau récapitulatif par lot
    table_lots_brut = (
        CIQ
        .dropna(subset=['lot_num', 'Date'])
        .groupby('lot_num', as_index=False)
        .agg(
            Date_min=('Date', 'min'),
            Date_max=('Date', 'max'),
            Nb_mesures=('Date', 'count')
        )
        .sort_values('Date_min')
    )

    # Format des dates pour l'affichage
    table_lots_brut['Date_min'] = table_lots_brut['Date_min'].dt.strftime('%Y-%m-%d')
    table_lots_brut['Date_max'] = table_lots_brut['Date_max'].dt.strftime('%Y-%m-%d')

    st.dataframe(
        table_lots_brut,
        width='stretch'
    )

    st.subheader("CV maximum utilisés")
    st.dataframe(df_cv_max, hide_index = True)
    st.write("CV max = CVa de la table EFLM")

    with st.expander("📊 Synthèse : choix des CV max"):

        # Nom du fichier sur votre serveur
        file_path = "10.1515_cclm-2024-0108.pdf"

        with open(file_path, "rb") as f:
            st.download_button(
                label="📥 Télécharger la référence (EFLM 2024 : Clin Chem Lab Med 2024; 62(8): 1483–1489)",
                data=f,
                file_name="Analytical performance specifications based on biological variation data - considerations, strengths and limitations",
                mime="application/pdf"
            )


    st.subheader("Recommandations EFLM + Probioqual ")
    # === import fichier excel EFLM_2025.xlsx ===
    # Charger la première feuille en DataFrame
    df_reco_eflm = pd.read_excel("EFLM_2025.xlsx", sheet_name=0, usecols=range(18))

    st.dataframe(df_reco_eflm,hide_index = True)
    st.markdown("#### Incertitudes de mesures recommandées: ")
    st.write(" - EFLM : privilégier MAU par rapport à TEa")
    st.write(" - PBQ : valeurs issues de l'ancienne table RICOS (TEa desirable)")
    st.write(" NB : tables RICOS remplacées par EFLM")
    st.write("[Site Officiel EFLM](https://biologicalvariation.eu/)")

    # Rappel des définitions

    st.markdown("### Légendes")
    st.write(r"CVI = \% Within-subject (CVI) estimate")
    st.write(r"CVG = \% Between-subject (CVG) estimate")
    st.write(r"$\text{MAU} = k \times \text{MAu} \quad (k=2)$")
    
    st.image("EFLM_definitions.png", caption="Définitions de l'EFLM")

with tab_CV_intralot:

    st.subheader("Calcul des CV intra-lot, par paramètre, par analyseur, par niveau de lot")

    # st.subheader("CV max - Fournisseur / Recommandé ")
    # Afficher un aperçu du DataFrame
    # st.dataframe(df_cv_max)

    # == Choix du numéro de lot ===

    st.subheader("Lots présents dans le jeu de données")

    # Tableau récapitulatif par lot
    table_lots = (
        CIQ
        .dropna(subset=['lot_num', 'Date'])
        .groupby('lot_num', as_index=False)
        .agg(
            Date_min=('Date', 'min'),
            Date_max=('Date', 'max'),
            Nb_mesures=('Date', 'count')
        )
        .sort_values('Date_min')
    )

    # Format des dates pour l'affichage
    table_lots['Date_min'] = table_lots['Date_min'].dt.strftime('%Y-%m-%d')
    table_lots['Date_max'] = table_lots['Date_max'].dt.strftime('%Y-%m-%d')

    st.dataframe(
        table_lots,
        width='stretch'
    )


    # lots_disponibles = sorted(CIQ['lot_num'].dropna().astype(str).unique())
    lots_disponibles = CIQ['lot_num'].astype(str).unique()
    filt_lot = st.selectbox("Numéro(s) de lot", lots_disponibles)

    # === Choix analyseurs ===

    filt_automate = st.multiselect("Automate(s)", sorted(CIQ[col_automate].dropna().unique()), default=None)

    # === Choix niveau de lot ===
    # Forcer tout en chaînes pour uniformiser les types
    niveaux_disponibles = sorted(CIQ['lot_niveau'].dropna().astype(str).unique())
    # Définir les niveaux souhaités par défaut (aussi en str)
    niveaux_defaut_souhaites = ['1101', '1102', '1103']
    # Ne garder que les niveaux par défaut présents dans les options
    niveaux_defaut_valides = [niveau for niveau in niveaux_defaut_souhaites if niveau in niveaux_disponibles]
    # Affichage du multiselect sécurisé
    filt_niveau = st.multiselect("Niveau(x) de lot", niveaux_disponibles, default=niveaux_defaut_valides)



    filt_annee = st.multiselect("Année(s)", sorted(CIQ['Annee'].dropna().unique()), default=None)


    # Filtrage des données
    data_filtrée = CIQ.copy()
    if filt_automate:
        data_filtrée = data_filtrée[data_filtrée[col_automate].isin(filt_automate)]
    if filt_niveau:
        data_filtrée = data_filtrée[data_filtrée['lot_niveau'].isin(filt_niveau)]
    if filt_lot:
        data_filtrée = data_filtrée[data_filtrée['lot_num'] == filt_lot]
    if filt_annee:
        data_filtrée = data_filtrée[data_filtrée['Annee'].isin(filt_annee)]

    st.subheader(f"Choix du paramètre à étudier pour le lot {filt_lot}")

    # === Choix du paramètre ===
    choix_param = CIQ.columns[8:]  # adapter si besoin
    param = st.selectbox("Choisissez le paramètre à étudier", choix_param)

    # Conversion du paramètre sélectionné en float
    data_filtrée[param] = pd.to_numeric(data_filtrée[param], errors='coerce')

    st.subheader(f"Tableau des CV intra-lot (CV classique / CV IQR / CV IQR robuste / CV MAD) pour {param} (lot {filt_lot})")

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

    # st.dataframe(grouped)

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


    st.dataframe(grouped, hide_index = True)
    st.write("Choix du CV max => cf Onlget Data source")

    # Rappel des formules utilisées pour le calcul du CV
    with st.expander("📊 Synthèse : Rappel des formules de CV utilisées"):
        st.markdown("### Formule du CV Classique")
        st.latex(r"CV_{classique} (\%) = \frac{\sigma}{\mu}*100")

        st.info(r"Où $\sigma$ représente l'écart-type de la série et $\mu$ représente la moyenne de la série.")

        st.markdown("### Formule du CV IQR (interquartile standard)")
        st.latex(r"CV_{IQR} (\%) = \frac{\text{IQR}}{\tilde{x}}*100")

        st.info(r"Où $\tilde{x}$ représente la médiane de la série et IQR représente l'intervalle interquartile (25%-75%).")

        st.markdown("### Formule du CV IQR_robuste (interquartile normalisé)")
        st.latex(r"CV_{IQR robuste} (\%) = \frac{\text{IQR}}{1,349*\tilde{x}}*100")

        st.info(r"Où $\tilde{x}$ représente la médiane de la série et IQR représente l'intervalle interquartile (25%-75%). Normalisation à la loi normale standard par le facteur 1,349.")

        st.markdown("### Formule du CV MAD (Median Absolute Deviation)")
        st.latex(r"CV_{MAD} (\%) = \frac{\text{median}(|x_i - \tilde{x}|)}{\tilde{x}}*1,4826*100")

        st.info(r"Où $\tilde{x}$ représente la médiane de la série. Normalisation à la loi normale standard par le facteur 1,4826. Tanterdtid, J., et al. (2007). Robustness of the median and the mean absolute deviation for the quality control of hematology analyzers.")

    with st.expander("📊 Synthèse : Avantages et Inconvénients des 4 méthodes"):
    
        st.markdown(r"#### 1. CV Classique ($\sigma/\mu$)")
        st.write("**Avantages :** Standard historique, connu de tous les biologistes et auditeurs (accréditation).")
        st.write("**Inconvénients :** Très sensible aux valeurs extrêmes (ex: fausse macrocytose). Risque de rejet de CIQ injustifié.")
        
        st.divider()

        st.markdown("#### 2. CV IQR (Standard)")
        st.write("**Avantages :** Simple, mesure le cœur de la population (50% centraux).")
        st.write("**Inconvénients :** Difficile à comparer aux limites de performance usuelles (valeurs numériques différentes).")
        
        st.divider()

        st.markdown("#### 3. CV IQR robuste (Normalisé par 1,349)")
        st.success("**Recommandé en Hématologie**")
        st.write("**Avantages :** Estime l'écart-type sur une distribution normale sans être pollué par les débris ou amas.")
        st.write("**Pourquoi ?** Donne le même chiffre que le CV classique si la distribution est propre.")
        
        st.divider()

        st.markdown("#### 4. CV MAD (Normalisé par 1,482)")
        st.write("**Avantages :** Statistique la plus robuste. Idéale pour les populations très bruitées.")
        st.write("**Inconvénients :** Parfois 'trop stable', peut masquer une dérive précoce. Plus complexe à justifier en audit (ISO 15189).")
    
        st.divider()

        st.markdown("""
        | Méthode | Robustesse | Sensibilité aux Outliers | Usage recommandé en Hématologie |
        | :--- | :---: | :---: | :--- |
        | **CV Classique** | ❌ Nulle | 🔴 Très sensible | Uniquement sur des populations parfaitement normales et propres. |
        | **CV IQR** | ✅ Bonne | 🟢 Très faible | Étude de la largeur de distribution (ex: RDW). |
        | **CV IQR robuste** | ✅ Excellente | 🟢 Faible | **Le meilleur compromis** pour comparer au CV classique cible. |
        | **CV MAD** | 🏆 Maximale | 🟢 Quasi nulle | Analyse de populations cellulaires très bruitées (Cytométrie/Hématologie). |
        """)

    # Graphs des CV intra-lot
    st.subheader(f"Graphiques des CV pour le lot {filt_lot}")
    plot_cv("CV", f"{param} : CV classique", "CV (%)")
    plot_cv("CV_IQR", f"{param} : CV IQR", "CV (%)")
    plot_cv("CV_IQR2", f"{param} : CV IQR robuste", "CV (%)")
    plot_cv("CV_MAD", f"{param} : CV MAD", "CV (%)")


    # =======================
    # ➕ Graphique Facets (CV_MAD par paramètre)
    # =======================


    st.subheader(f"Comparaison des CV intra-lot (méthode MAD) pour les paramètres du lot {filt_lot}")


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
        "Paramètres à afficher",
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

    # Sélecteurs pour le graphique
    st.subheader("Visualisation de la carte de contrôle (Levey-Jennings)")
    # On n'a plus besoin de choisir le paramètre (il est déjà dans param_brut)
    # Mais on doit choisir l'Automate et le Niveau si plusieurs sont présents dans les données filtrées
    col1, col2 = st.columns(2)
    with col1:
        # On choisit l'automate (Nickname) parmi ceux restants après filtrage
        automate_choisi = st.selectbox("Automate à visualiser (intralot):", data_filtrée[col_automate].unique())
    with col2:
        # On choisit le niveau parmi ceux restants
        niveau_choisi = st.selectbox("Niveau à visualiser (intralot):", data_filtrée['lot_niveau'].unique())

    # --- FILTRAGE DES DONNÉES ---
    # On prend les lignes correspondant à l'automate et au niveau
    df_plot = data_filtrée[
        (data_filtrée[col_automate] == automate_choisi) & 
        (data_filtrée['lot_niveau'] == niveau_choisi)
    ].copy()

    # --- RÉCUPÉRATION DES STATS ---
    # Attention : on filtre grouped_brut sur l'automate et le niveau pour avoir la moyenne/SD
    stats_selection = grouped[
        (grouped[col_automate] == automate_choisi) & 
        (grouped['lot_niveau'] == niveau_choisi)
    ]

    if not stats_selection.empty:
        stats = stats_selection.iloc[0]
        moy_brut = stats['Moyenne']
        sd_brut = stats['Ecart_type']
        cv_brut = stats['CV']
        
        # Appel de la fonction de graphique (en utilisant df_plot[param_brut])
        # st.plotly_chart(plot_levey_jennings(df_plot, moy_brut, sd_brut, param_brut))
    else:
        st.warning("Pas de statistiques calculées pour cette sélection.")

    def generer_levey_jennings(df, param_nom, moyenne, sd):
        # Tri par date pour un tracé chronologique
        df = df.sort_values('Date')
        
        fig = go.Figure()

        # Définition des zones de contrôle (±1SD, ±2SD, ±3SD)
        limites = {
            'Moyenne': {'val': moyenne, 'color': 'green', 'dash': 'solid'},
            '+1 SD': {'val': moyenne + sd, 'color': 'orange', 'dash': 'dot'},
            '-1 SD': {'val': moyenne - sd, 'color': 'orange', 'dash': 'dot'},
            '+2 SD': {'val': moyenne + 2*sd, 'color': 'red', 'dash': 'dash'},
            '-2 SD': {'val': moyenne - 2*sd, 'color': 'red', 'dash': 'dash'},
            '+3 SD': {'val': moyenne + 3*sd, 'color': 'darkred', 'dash': 'dashdot'},
            '-3 SD': {'val': moyenne - 3*sd, 'color': 'darkred', 'dash': 'dashdot'},
        }

        for label, config in limites.items():
            fig.add_hline(y=config['val'], 
                        line=dict(color=config['color'], dash=config['dash'], width=1),
                        annotation_text=label, 
                        annotation_position="top right")

        # Ajout des points de mesure
        # On colorie les points dynamiquement selon leur éloignement
        colors = []
        for val in df[param_nom]:
            if abs(val - moyenne) > 3 * sd: 
                colors.append('darkred')
            elif abs(val - moyenne) > 2 * sd: 
                colors.append('red')
            else: 
                colors.append('blue')

        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df[param_nom],
            mode='lines+markers',
            name=param_nom,
            line=dict(color='lightgray', width=1),
            marker=dict(size=10, color=colors, symbol='circle')
        ))

        fig.update_layout(
            title=f"Levey-Jennings : {param_nom} (Lot: {df['lot_num'].iloc[0]})",
            xaxis_title="Date d'analyse",
            yaxis_title="Valeur mesurée",
            template="plotly_white",
            height=600
        )
        
        return fig

# --- SELECTION POUR LE GRAPHIQUE ---

    if not data_filtrée.empty:
        col_g1, col_g2 = st.columns(2)
        
        # with col_g1:
        #     # On choisit l'automate parmi ceux présents dans les données filtrées
        #     automate_choisi = st.selectbox("Sélectionner l'automate :", data_filtrée_brut[col_automate].unique())
        
        # with col_g2:
        #     # On choisit le niveau
        #     niveau_choisi = st.selectbox("Sélectionner le niveau :", data_filtrée_brut['lot_niveau'].unique())

        # --- FILTRAGE FINAL POUR LE GRAPH ---
        df_plot = data_filtrée[
            (data_filtrée[col_automate] == automate_choisi) & 
            (data_filtrée['lot_niveau'] == niveau_choisi)
        ].copy()

        # Récupération des stats calculées précédemment dans grouped
        # Rappel : grouped contient une ligne par (Automate, Niveau, Annee)
        stats_select = grouped[
            (grouped[col_automate] == automate_choisi) & 
            (grouped['lot_niveau'] == niveau_choisi)
        ]

        if not stats_select.empty and not df_plot.empty:
            # On prend les stats de la première ligne correspondante
            s = stats_select.iloc[0]
            
            # Génération du graphique
            fig_lj = generer_levey_jennings(
                df_plot, 
                param,      # Le nom de la colonne choisie plus haut
                s['Moyenne'], 
                s['Ecart_type']
            )
            
            st.plotly_chart(fig_lj, width = 'stretch', key="LJ intralot")
            
            # Petit récapitulatif sous le graph
            st.info(f"**Statistiques pour ce graphique :** n={s['n']} | Moyenne={s['Moyenne']:.3f} | SD={s['Ecart_type']:.3f} | CV={s['CV']:.3f}")
        else:
            st.warning("Données insuffisantes pour générer le graphique sur cette sélection.")
    else:
        st.error("Le jeu de données filtré est vide.")



with tab_CV_interlot:

    # == Choix du numéro de lot ===

    st.subheader("Lots présents dans le jeu de données")

    # Tableau récapitulatif par lot
    table_lots_cvinter = (
        CIQ
        .dropna(subset=['lot_num', 'Date'])
        .groupby('lot_num', as_index=False)
        .agg(
            Date_min=('Date', 'min'),
            Date_max=('Date', 'max'),
            Nb_mesures=('Date', 'count')
        )
        .sort_values('Date_min')
    )

    # Format des dates pour l'affichage
    table_lots_cvinter['Date_min'] = table_lots_cvinter['Date_min'].dt.strftime('%Y-%m-%d')
    table_lots_cvinter['Date_max'] = table_lots_cvinter['Date_max'].dt.strftime('%Y-%m-%d')

    st.dataframe(
        table_lots_cvinter,
        width='stretch'
    )


    lots_disponibles_cvinter = sorted(CIQ['lot_num'].dropna().astype(str).unique())
    filt_lot_cvinter = st.multiselect("Numéro(s) de lot (CV inter-lot)", lots_disponibles_cvinter)


    # === Choix analyseurs ===

    filt_automate_cvinter = st.multiselect("Automate(s) (CV inter-lot)", sorted(CIQ[col_automate].dropna().unique()), default=None)

    # === Choix niveau de lot ===
    # Forcer tout en chaînes pour uniformiser les types
    niveaux_disponibles_cvinter = sorted(CIQ['lot_niveau'].dropna().astype(str).unique())
    # Définir les niveaux souhaités par défaut (aussi en str)
    niveaux_defaut_souhaites_cvinter = ['1101', '1102', '1103']
    # Ne garder que les niveaux par défaut présents dans les options
    niveaux_defaut_valides_cvinter = [niveau for niveau in niveaux_defaut_souhaites_cvinter if niveau in niveaux_disponibles_cvinter]
    # Affichage du multiselect sécurisé
    filt_niveau_cvinter = st.multiselect("Niveau(x) de lot (CV inter-lot)", niveaux_disponibles_cvinter, default=niveaux_defaut_valides_cvinter)



    filt_annee_cvinter = st.multiselect("Année(s) (CV inter-lot)", sorted(CIQ['Annee'].dropna().unique()), default=None)

    # Filtrage des données
    data_filtrée_cvinter = CIQ.copy()
    if filt_automate_cvinter:
        data_filtrée_cvinter = data_filtrée_cvinter[data_filtrée_cvinter[col_automate].isin(filt_automate_cvinter)]
    if filt_niveau_cvinter:
        data_filtrée_cvinter = data_filtrée_cvinter[data_filtrée_cvinter['lot_niveau'].isin(filt_niveau_cvinter)]
    if filt_lot_cvinter:
        data_filtrée_cvinter = data_filtrée_cvinter[data_filtrée_cvinter['lot_num'].isin(filt_lot_cvinter)]
    if filt_annee_cvinter:
        data_filtrée_cvinter = data_filtrée_cvinter[data_filtrée_cvinter['Annee'].isin(filt_annee_cvinter)]

    st.subheader(f"Choix du paramètre à étudier pour les lots {filt_lot}")

    # === Choix du paramètre ===
    choix_param_cvinter = CIQ.columns[8:]  # adapter si besoin
    param_cvinter = st.selectbox("Choisissez le paramètre à étudier (CV inter-lot)", choix_param_cvinter)

    # Conversion du paramètre sélectionné en float
    data_filtrée_cvinter[param_cvinter] = pd.to_numeric(data_filtrée_cvinter[param_cvinter], errors='coerce')

    # Agrégation par automate, lot_num et niveau
    grouped_cvinter = data_filtrée_cvinter.groupby([col_automate, 'lot_num','lot_niveau','Annee'])[param_cvinter].agg(
        n='count',
        Moyenne='mean',
        Mediane='median',
        Ecart_type='std',
        CV=cv,
        # CV_IQR=cv_robuste_iqr,
        CV_IQR2=cv_robuste_iqr2,
        CV_MAD=cv_robuste_mad
    ).reset_index()

    grouped_cvinter['paramètre'] = param_cvinter

    # 1. Conversion forcée en string pour garantir la correspondance
    grouped_cvinter['lot_niveau'] = grouped_cvinter['lot_niveau'].astype(str).str.strip()
    df_cv_max['lot_niveau'] = df_cv_max['lot_niveau'].astype(str).str.strip()

    # 2. On fait de même pour la colonne 'paramètre' par sécurité
    grouped_cvinter['paramètre'] = grouped_cvinter['paramètre'].astype(str).str.strip()
    df_cv_max['paramètre'] = df_cv_max['paramètre'].astype(str).str.strip()

    # 3. Maintenant le merge fonctionnera
    grouped_cvinter = grouped_cvinter.merge(
        df_cv_max[['paramètre', 'lot_niveau', 'CV_max_reco']], 
        on=['paramètre', 'lot_niveau'], 
        how='left'
    )



    st.subheader(f"Tableau des CV inter-lot (CV classique / CV IQR / CV IQR robuste / CV MAD) par Lot pour {param_cvinter} (lot(s) {filt_lot_cvinter})")
    st.dataframe(grouped_cvinter, hide_index = True)

    grouped_cvinter['lot_annee'] = grouped_cvinter['lot_niveau'].astype(str) + " (" + grouped_cvinter['Annee'].astype(str) + ")"
    grouped_cvinter['lot_num2'] = grouped_cvinter['lot_num'].astype(str) + " (" + grouped_cvinter['lot_niveau'].astype(str) + ")"

    st.subheader(f"Graphiques des CV inter-lot (lot(s) {filt_lot_cvinter})")
    plot_cvinter("CV", f"{param_cvinter} : CV classique", "CV (%)")
    # plot_cvinter("CV_IQR", f"{param_cvinter} : CV IQR", "CV (%)")
    plot_cvinter("CV_IQR2", f"{param_cvinter} : CV IQR robuste", "CV (%)")
    plot_cvinter("CV_MAD", f"{param_cvinter} : CV MAD", "CV (%)")


with tab_CVref:

    st.subheader("Calcul des CV de référence")

 # == Choix du numéro de lot ===

    st.subheader("Lots présents dans le jeu de données")

    # Tableau récapitulatif par lot
    table_lots_cvref = (
        CIQ
        .dropna(subset=['lot_num', 'Date'])
        .groupby('lot_num', as_index=False)
        .agg(
            Date_min=('Date', 'min'),
            Date_max=('Date', 'max'),
            Nb_mesures=('Date', 'count')
        )
        .sort_values('Date_min')
    )

    # Format des dates pour l'affichage
    table_lots_cvref['Date_min'] = table_lots_cvref['Date_min'].dt.strftime('%Y-%m-%d')
    table_lots_cvref['Date_max'] = table_lots_cvref['Date_max'].dt.strftime('%Y-%m-%d')

    st.dataframe(
        table_lots_cvref,
        width='stretch'
    )


    lots_disponibles_cvref = sorted(CIQ['lot_num'].dropna().astype(str).unique())
    filt_lot_cvref = st.multiselect("Numéro(s) de lot (CV ref)", lots_disponibles_cvref)

    # === Choix analyseurs ===

    filt_automate_cvref = st.multiselect("Automate(s) (CV ref)", sorted(CIQ[col_automate].dropna().unique()), default=None)

    # === Choix niveau de lot ===
    # Forcer tout en chaînes pour uniformiser les types
    niveaux_disponibles_cvref = sorted(CIQ['lot_niveau'].dropna().astype(str).unique())
    # Définir les niveaux souhaités par défaut (aussi en str)
    niveaux_defaut_souhaites_cvref = ['1101', '1102', '1103']
    # Ne garder que les niveaux par défaut présents dans les options
    niveaux_defaut_valides_cvref = [niveau for niveau in niveaux_defaut_souhaites_cvref if niveau in niveaux_disponibles_cvref]
    # Affichage du multiselect sécurisé
    filt_niveau_cvref = st.multiselect("Niveau(x) de lot (CV ref)", niveaux_disponibles_cvref, default=niveaux_defaut_valides_cvref)

    filt_annee_cvref = st.multiselect("Année(s) (CV ref)", sorted(CIQ['Annee'].dropna().unique()), default=None)

    # Filtrage des données
    data_filtrée_cvref = CIQ.copy()
    if filt_automate_cvref:
        data_filtrée_cvref = data_filtrée_cvref[data_filtrée_cvref[col_automate].isin(filt_automate_cvref)]
    if filt_niveau_cvref:
        data_filtrée_cvref = data_filtrée_cvref[data_filtrée_cvref['lot_niveau'].isin(filt_niveau_cvref)]
    if filt_lot_cvref:
        data_filtrée_cvref = data_filtrée_cvref[data_filtrée_cvref['lot_num'].isin(filt_lot_cvref)]
    if filt_annee_cvref:
        data_filtrée_cvref = data_filtrée_cvref[data_filtrée_cvref['Annee'].isin(filt_annee_cvref)]


    st.subheader(f"Tableau des CV de référence (CV classique / CV IQR / CV IQR robuste / CV MAD / CV poolé robuste) | lots {filt_lot_cvref}")

    # === Choix du paramètre ===
    choix_param_cvref = CIQ.columns[8:]  # adapter si besoin
    param_cvref = st.selectbox("Choisissez le paramètre à étudier (CV ref)", choix_param_cvref)

    # Conversion du paramètre sélectionné en float
    data_filtrée_cvref[param_cvref] = pd.to_numeric(data_filtrée_cvref[param_cvref], errors='coerce')

    # st.dataframe(data_filtrée_cvref)

    # --- CALCUL DU CV POOLED ROBUSTE ---
    # On utilise apply pour passer par lot_num tout en restant groupé par Automate/Niveau/Annee
    cv_pooled_rob_df = data_filtrée_cvref.groupby([col_automate, 'lot_niveau', 'Annee']).apply(
        lambda x: calculate_cv_pooled_robust_internal(x, param_cvref, 'lot_num')
    ).reset_index(name='CV_Pooled_Robuste')

    # Agrégation par automate, lot_num et niveau
    grouped_cvref = data_filtrée_cvref.groupby([col_automate,'lot_niveau','Annee'])[param_cvref].agg(
        n='count',
        Moyenne='mean',
        Mediane='median',
        Ecart_type='std',
        CV=cv,
        # CV_IQR=cv_robuste_iqr,
        CV_IQR2=cv_robuste_iqr2,
        CV_MAD=cv_robuste_mad
    ).reset_index()

    # --- FUSION DES DEUX ---
    grouped_cvref = grouped_cvref.merge(
        cv_pooled_rob_df, 
        on=[col_automate, 'lot_niveau', 'Annee'], 
        how='left'
    )

    grouped_cvref['paramètre'] = param_cvref

    st.dataframe(grouped_cvref, hide_index = True)

    # Affichage de la formule pour justifier le calcul
    with st.expander("🔬 Note méthodologique : CV Pooled Robuste"):
        st.latex(r"CV_{pooled\_rob} = \frac{\sqrt{\frac{\sum (n_i - 1) \cdot (MAD_i \cdot 1.4826)^2}{\sum n_i - k}}}{\mu_{globale}} \times 100")
        st.write("Ce CV est calculé en combinant les variances robustes de chaque numéro de lot sélectionné. Il neutralise l'effet des sauts de moyennes entre les lots.")
        st.info("**Justification pour l'accréditation (ISO 15189)**")
        st.markdown(r"""
        **Objet : Méthodologie de calcul des limites de performance.**
        Afin de garantir une estimation fiable et représentative de la précision de nos méthodes d'analyse, le laboratoire a fait le choix d'utiliser le Coefficient de Variation (CV) poolé robuste pour le suivi de ses indicateurs de qualité et le calcul de l’incertitude de mesure (conformément à l'article 7.3.3 de l'ISO 15189).
        Cette approche repose sur la combinaison des variances de plusieurs lots de contrôle (pooling), permettant ainsi d'intégrer la variabilité interlot et d'augmenter la puissance statistique de nos estimations. Pour s'affranchir de l'influence indue des valeurs aberrantes ou des incidents analytiques isolés, une méthode de robustesse (type élimination des outliers ou utilisation de la MAD) est appliquée systématiquement.
        Ce choix méthodologique assure la stabilité des limites de contrôle dans le temps, évite la redéfinition erratique des cibles à chaque changement de lot et permet une surveillance fine des dérives analytiques réelles, garantissant ainsi la validité clinique des résultats délivrés.
        * **Robustesse :** Application du facteur $1.4826 \times MAD$.
        * **Stabilité :** Neutralisation des sauts de moyennes interlots.
        """)
        
    st.subheader(f"Comparaison des CV de référence (CV MAD) pour les paramètres sélectionnés | lots {filt_lot_cvref}")

    # Sélectionne les colonnes de l'index 8 à 125 pour permettre la conversion en numérique
    # st.dataframe(CIQ.head())
    colonnes_numeriques_cvref = CIQ.columns[8:125]

    # Nettoyage et conversion en float
    for col in colonnes_numeriques_cvref:
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
    params_numeriques_cvref = CIQ.select_dtypes(include=[np.number]).columns.tolist()

    # Exclure les colonnes de comptage ou de type ID si nécessaire
    params_numeriques_cvref = [col for col in params_numeriques_cvref if col not in ['n']]

    # Liste par défaut
    params_visibles_par_défaut_cvref = [    
        'WBC(10^9/L)','RBC(10^12/L)','HGB(g/L)','HCT(%)','MCV(fL)','MCH(pg)','MCHC(g/L)','PLT(10^9/L)','[RBC-O(10^12/L)]','[PLT-O(10^9/L)]','[PLT-F(10^9/L)]','IPF#(10^9/L)','[HGB-O(g/dL)]'
        ]

    # Sélecteur des paramètres à inclure dans les facets
    params_selectionnés_cvref = st.multiselect(
        "Paramètres à sélectionner (CV ref)",
        options=params_numeriques_cvref,
        default=[p for p in params_visibles_par_défaut_cvref if p in params_numeriques_cvref]
    )

    ordre_parametres = params_selectionnés_cvref


    # pour attribuer toujours la même couleur aux analyseurs
    # 1. Identifier tous les automates uniques
    automates_uniques = sorted(data_filtrée_cvref[col_automate].unique())

    # 2. Créer un dictionnaire de couleurs (ex: Bleu, Rouge, Vert...)
    # On utilise une palette qualitative de Plotly (ex: Set1, D3, Plotly)
    palette = px.colors.qualitative.Plotly 
    automates_color_map = {automate: palette[i % len(palette)] for i, automate in enumerate(automates_uniques)}
    # Color map perso si besoin : Remplacez "Automate_1", etc. par les noms exacts présents dans votre colonne col_automate
    # color_map_perso = {
    #     "XN-1000": "#1f77b4",  # Un bleu spécifique
    #     "XN-2000": "#FF4B4B",  # Le rouge Streamlit
    #     "XN-3000": "orange",   # Nom de couleur standard
    #     "Lieu_A": "#2ca02c"    # Un vert
    # }

    #st.write("Paramètres dans Excel :", df_cv_max['paramètre'].unique())
    #st.write("Paramètres dans vos données :", params_selectionnés)

    # Compilation des CV_MAD pour chaque paramètre
    liste_dfs_cvref = []

    for p_cvref in params_selectionnés_cvref:
        df_tmp_cvref = data_filtrée_cvref.groupby([col_automate, 'lot_niveau'])[p_cvref].agg(
            n='count',
            CV_MAD=cv_robuste_mad
        ).reset_index()
        
        # On ajoute le nom du paramètre pour pouvoir faire le merge
        df_tmp_cvref['paramètre'] = p_cvref
        df_tmp_cvref.rename(columns={'CV_MAD': 'CV'}, inplace=True)
             
        liste_dfs_cvref.append(df_tmp_cvref)



    # Fusion de tous les DataFrames
    df_facet_cvref = pd.concat(liste_dfs_cvref, ignore_index=True)

    # On s'assure que le DataFrame est propre
    df_facet_cvref['lot_niveau'] = df_facet_cvref['lot_niveau'].astype(str)

    # Conversion en catégorie avec l'ordre défini
    df_facet_cvref['paramètre'] = pd.Categorical(
        df_facet_cvref['paramètre'], 
        categories=ordre_parametres, 
        ordered=True
    )

    # Tri du DataFrame pour s'assurer que Plotly respecte l'ordre
    df_facet_cvref = df_facet_cvref.sort_values('paramètre')

    # Graphique facets avec Plotly Express
    fig_facet_cvref = px.bar(
        df_facet_cvref,
        x='lot_niveau',
        y='CV',
        color=col_automate,
        color_discrete_map=automates_color_map,  # Force l'utilisation du dictionnaire
        category_orders={"paramètre": ordre_parametres, col_automate: automates_uniques},   
        barmode='group',
        facet_col='paramètre',
        facet_col_wrap=3,  # Nombre de colonnes dans la grille
        title='CV MAD par paramètre et par niveau de lot',
        facet_row_spacing=0.1,
        facet_col_spacing=0.1,
        height=1500,  # Plus grand pour laisser de la place
        labels={'CV': 'CV MAD (%)', 'lot_niveau': 'Niveau de lot'}
    )

    fig_facet_cvref.update_yaxes(matches=None)  # axes Y indépendants


    # Affiche tous les labels d’axe Y
    for axis in fig_facet_cvref.layout:
        if axis.startswith("yaxis"):
            fig_facet_cvref.layout[axis].showticklabels = True
            fig_facet_cvref.layout[axis].title = dict(text="CV MAD (%)")


    # Forcer l’affichage de l’axe X et du titre sur chaque subplot
    for axis_name in fig_facet_cvref.layout:
        if axis_name.startswith("xaxis"):
            axis = fig_facet_cvref.layout[axis_name]
            axis.showticklabels = True  # Affiche les ticks
            axis.title = dict(text="Niveau de lot")  # Titre de l'axe X


    fig_facet_cvref.update_layout(height=300 * ((len(params_selectionnés_cvref) - 1) // 3 + 1))  # ajuste la hauteur automatiquement
    st.plotly_chart(fig_facet_cvref, width='stretch')

    st.subheader(f"Comparaison des CV de référence (CV poolé robuste) pour les paramètres sélectionnés | lots {filt_lot_cvref}")

    # Compilation des CV pooled robuste pour chaque paramètre
    liste_dfs_cvref2 = []

    for p in params_selectionnés_cvref:
        # On effectue le calcul pour chaque paramètre du multiselect
        df_p = data_filtrée_cvref.groupby([col_automate, 'lot_niveau', 'Annee']).apply(
            lambda x: calculate_cv_pooled_robust_internal(x, p, 'lot_num'),
            include_groups=False
        ).reset_index(name='CV')
        
        df_p['paramètre'] = p
        liste_dfs_cvref2.append(df_p)

    # On fusionne tout pour le graphique facet
    if liste_dfs_cvref2:
        df_facet_cvref2 = pd.concat(liste_dfs_cvref2, ignore_index=True)
    else:
        st.warning("Aucune donnée disponible pour les paramètres sélectionnés.")
    
    # On s'assure qu'on a une seule valeur unique par combinaison (Barre)
    df_plot = df_facet_cvref2.groupby([col_automate, 'lot_niveau', 'paramètre'], as_index=False).agg({'CV': 'mean'})

    # Nettoyage final des types
    df_plot['CV'] = pd.to_numeric(df_plot['CV'], errors='coerce')
    df_plot['lot_niveau'] = df_plot['lot_niveau'].astype(str)

    # Conversion en catégorie avec le même ordre
    df_plot['paramètre'] = pd.Categorical(
        df_plot['paramètre'], 
        categories=ordre_parametres, 
        ordered=True
    )

    # Tri du DataFrame
    df_plot = df_plot.sort_values('paramètre')

    fig_facet_cvref2 = px.bar(
        df_plot, # Utilisation du DF agrégé
        x='lot_niveau',
        y='CV',
        color=col_automate,
        color_discrete_map=automates_color_map,  # Force l'utilisation du dictionnaire
        category_orders={"paramètre": ordre_parametres, col_automate: automates_uniques},
        barmode='group',
        facet_col='paramètre',
        facet_col_wrap=3,
        title='CV poolé robuste par paramètre et par niveau de lot',
        facet_row_spacing=0.1,
        facet_col_spacing=0.1,
        labels={'CV': 'CV (%)', 'lot_niveau': 'Niveau'}
    )

    # 1. Libérer les axes Y
    fig_facet_cvref2.update_yaxes(matches=None, showticklabels=True)

    # 2. Forcer les titres et le comportement des axes pour chaque subplot
    for axis in fig_facet_cvref2.layout:
        if axis.startswith("yaxis"):
            fig_facet_cvref2.layout[axis].update(
                title_text="CV poolé robuste (%)",
                showticklabels=True,
                rangemode="tozero" # Force l'axe à partir de 0
            )
        if axis.startswith("xaxis"):
            fig_facet_cvref2.layout[axis].update(
                title_text="Niveau",
                showticklabels=True
            )

    # 3. Ajustement de la hauteur
    n_rows = math.ceil(len(params_selectionnés_cvref) / 3)
    fig_facet_cvref2.update_layout(height=350 * n_rows)

    st.plotly_chart(fig_facet_cvref2, width='stretch')

    # =======================
    # ➕ Graphique Facets ( valeur paramètre) avec filtre année
    # =======================

    st.subheader("Distribution des valeurs de chaque paramètre")

    if st.button("Afficher la distribution des valeurs des paramètres"):
        
        if len(params_selectionnés_cvref) == 0:
            st.warning("Veuillez sélectionner au moins un paramètre.")
        else:
            df_facet_cvref = data_filtrée_cvref[[col_automate, 'lot_niveau','Annee'] + params_selectionnés_cvref].copy()
            df_facet_cvref['lot_niveau'] = df_facet_cvref['lot_niveau'].astype(str)
            
        
            df_melted_cvref = df_facet_cvref.melt(
                id_vars=[col_automate, 'lot_niveau','Annee'],
                value_vars=params_selectionnés_cvref,
                var_name='paramètre',
                value_name='valeur'
            )
        
            df_melted_cvref[col_automate] = df_melted_cvref[col_automate].astype(str)
            df_melted_cvref['lot_niveau'] = df_melted_cvref['lot_niveau'].astype(str)
            df_melted_cvref = df_melted_cvref.dropna()
        
            
        
            fig_facet_cvref3 = px.box(
                df_melted_cvref,
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
            fig_facet_cvref3.update_yaxes(matches=None)
        
            for axis in fig_facet_cvref3.layout:
                if axis.startswith("yaxis"):
                    fig_facet_cvref3.layout[axis].showticklabels = True
                    fig_facet_cvref3.layout[axis].title = dict(text="Valeur")
        
            for axis_name in fig_facet_cvref3.layout:
                if axis_name.startswith("xaxis"):
                    axis = fig_facet_cvref3.layout[axis_name]
                    axis.showticklabels = True
                    axis.title = dict(text="Niveau de lot")
        
            fig_facet_cvref3.update_layout(height=300 * ((len(params_selectionnés_cvref) - 1) // 3 + 1))  # ajuste la hauteur automatiquement
            st.plotly_chart(fig_facet_cvref3, width='stretch')

    st.subheader("Distribution détaillée par Numéro de Lot")

    if st.button("Afficher la distribution par Lot"):
        if len(params_selectionnés_cvref) == 0:
            st.warning("Veuillez sélectionner au moins un paramètre.")
        else:
            # 1. Préparation des données incluant 'lot_num'
            # On ajoute 'lot_num' aux colonnes extraites
            df_facet_lot = data_filtrée_cvref[[col_automate, 'lot_niveau', 'lot_num', 'Annee'] + params_selectionnés_cvref].copy()
            
            # Conversion en string pour éviter les problèmes d'affichage
            df_facet_lot['lot_num'] = df_facet_lot['lot_num'].astype(str)
            df_facet_lot['lot_niveau'] = df_facet_lot['lot_niveau'].astype(str)

            # 2. Passage au format long (Melt)
            df_melted_lot = df_facet_lot.melt(
                id_vars=[col_automate, 'lot_niveau', 'lot_num', 'Annee'],
                value_vars=params_selectionnés_cvref,
                var_name='paramètre',
                value_name='valeur'
            ).dropna()

            # 3. Création du Boxplot
            # On met 'lot_num' en X pour voir l'évolution lot par lot
            fig_facet_lot = px.box(
                df_melted_lot,
                x='lot_num', 
                y='valeur',
                color=col_automate,
                facet_col='paramètre',
                facet_col_wrap=3,
                title="Distribution des paramètres par Numéro de Lot (Détail)",
                facet_row_spacing=0.08,
                facet_col_spacing=0.08,
                labels={
                    'valeur': 'Valeur',
                    'lot_num': 'N° de Lot',
                    col_automate: 'Automate'
                },
                # On peut ajouter les points pour mieux voir la dispersion
                points="outliers" 
            )

            # 4. Nettoyage des axes (Indépendance et labels)
            fig_facet_lot.update_yaxes(matches=None, showticklabels=True)
            fig_facet_lot.update_xaxes(showticklabels=True, tickangle=45) # Inclinaison car les numéros de lots sont longs

            for axis_name in fig_facet_lot.layout:
                if axis_name.startswith("yaxis"):
                    fig_facet_lot.layout[axis_name].title = dict(text="Valeur")
                if axis_name.startswith("xaxis"):
                    fig_facet_lot.layout[axis_name].title = dict(text="Lot")

            # 5. Ajustement dynamique de la hauteur
            n_rows = math.ceil(len(params_selectionnés_cvref) / 3)
            fig_facet_lot.update_layout(height=400 * n_rows)

            st.plotly_chart(fig_facet_lot, width='stretch')

with tab_IM:
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
    options = ["Importer un fichier EEQ", "Utiliser un fichier EEQ par défaut", "Rechercher un fichier EEQ en local"]
    choix_eeq = st.radio("Source du fichier EEQ :", options)

    if choix_eeq == "Importer un fichier EEQ":
        uploaded_eeq = st.file_uploader("Importer fichier EEQ", type=["csv"])
        if uploaded_eeq:
            EEQ = lire_fichier_eeq(contenu_brut=uploaded_eeq.read(), nom=uploaded_eeq.name)
            if EEQ is not None:
                st.success(f"Fichier importé : {uploaded_eeq.name}")
            else:
                st.stop()
        else:
            st.stop()

    elif choix_eeq == "Utiliser un fichier EEQ par défaut":
        EEQ = lire_fichier_eeq(fichier_path="exportEEQ1952.csv")
        if EEQ is not None:
            st.success("Fichier par défaut chargé.")
        else:
            st.stop()

    elif choix_eeq == "Rechercher un fichier EEQ en local":
        # On liste les fichiers CSV du dossier actuel (ou un chemin spécifique)
        fichiers_locaux = [f for f in os.listdir('.') if f.endswith('.csv') and 'EEQ' in f.upper()]
        
        if fichiers_locaux:
            fichier_choisi = st.selectbox("Sélectionnez un fichier EEQ trouvé en local :", fichiers_locaux)
            if fichier_choisi:
                EEQ = lire_fichier_eeq(fichier_path=fichier_choisi)
                st.success(f"Fichier local chargé : {fichier_choisi}")
        else:
            st.error("Aucun fichier contenant 'EEQ' n'a été trouvé dans le dossier local.")
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
    
    # Filtrer app NK9 ou NKR (après mi 2025)
    EEQ = EEQ[EEQ['App'].isin(['NK9', 'NKR'])]   

    # Ajouter Nickname en fonction date et Anonymat (adaptation directe de R -> Python)
    def assign_nickname(row):
        d = row['Date']
        a = row['Anonymat']
        if pd.isna(d) or pd.isna(a):
            return np.nan
        if d >= pd.Timestamp("2013-01-01") and d <= pd.Timestamp("2023-05-10"):
            if a == "1952": 
                return "ATHOS-1"
            elif a == "1952A": 
                return "PORTHOS-2"
            elif a == "1952B": 
                return "ARAMIS-3"
        elif d >= pd.Timestamp("2023-05-11") and d <= pd.Timestamp("2023-12-10"):
            if a == "1952": 
                return "XN-9100-1-A"
            elif a == "1952A": 
                return "XN-9100-2-A"
            elif a == "1952B": 
                return "XN-9100-3-A"
        elif d >= pd.Timestamp("2023-12-11"):
            if a == "1952": 
                return "XR-ISIS-A"
            elif a == "1952A": 
                return "XR-OSIRIS-A"
            elif a == "1952B": 
                return "XR-ANUBIS-A"
            elif a == "1952Z": 
                return "XN-1000-1-A"
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

    # Calcul du biais moyen absolu par groupe
    # st.dataframe(EEQ.head())

    EEQ = EEQ.rename(columns={"variable": "Paramètre"})

    if "Paramètre" in EEQ.columns:
        EEQ["Biais |c| pairs"] = (
        EEQ["Biais |c| pairs"]
        .str.replace(",", ".", regex=False)  # remplacer la virgule par un point
        .astype(float)                       # convertir en float
        )
    with st.expander("Data EEQ"):
        st.dataframe(EEQ, width='stretch')


    #### Choix des lots de CIQ pour calcul des IM #####
    ## choix sur année et/ou numéro de lot
    st.subheader('Choix des lots de CIQ à utiliser pour le calcul des IM')
    # Tableau récapitulatif par lot
    table_lots_IM = (
        CIQ
        .dropna(subset=['lot_num', 'Date'])
        .groupby('lot_num', as_index=False)
        .agg(
            Date_min=('Date', 'min'),
            Date_max=('Date', 'max'),
            Nb_mesures=('Date', 'count')
        )
        .sort_values('Date_min')
    )

    # Format des dates pour l'affichage
    table_lots_IM['Date_min'] = table_lots_IM['Date_min'].dt.strftime('%Y-%m-%d')
    table_lots_IM['Date_max'] = table_lots_IM['Date_max'].dt.strftime('%Y-%m-%d')

    st.dataframe(
        table_lots_IM,
        width='stretch'
    )


    lots_disponibles_IM = sorted(CIQ['lot_num'].dropna().astype(str).unique())
    filt_lot_IM = st.multiselect("Numéro(s) de lot (IM)", lots_disponibles_IM)

    # === Choix analyseurs ===

    filt_automate_IM = st.multiselect("Automate(s) (IM)", sorted(CIQ[col_automate].dropna().unique()), default=None)

    # === Choix niveau de lot ===
    # Forcer tout en chaînes pour uniformiser les types
    niveaux_disponibles_IM = sorted(CIQ['lot_niveau'].dropna().astype(str).unique())
    # Définir les niveaux souhaités par défaut (aussi en str)
    niveaux_defaut_souhaites_IM = ['1101', '1102', '1103']
    # Ne garder que les niveaux par défaut présents dans les options
    niveaux_defaut_valides_IM = [niveau for niveau in niveaux_defaut_souhaites_IM if niveau in niveaux_disponibles_IM]
    # Affichage du multiselect sécurisé
    filt_niveau_IM = st.multiselect("Niveau(x) de lot (IM)", niveaux_disponibles_IM, default=niveaux_defaut_valides_IM)

    filt_annee_IM = st.multiselect("Année(s) (IM)", sorted(CIQ['Annee'].dropna().unique()), default=None)

    # Filtrage des données
    data_filtrée_IM = CIQ.copy()
    if filt_automate_IM:
        data_filtrée_IM = data_filtrée_IM[data_filtrée_IM[col_automate].isin(filt_automate_IM)]
    if filt_niveau_IM:
        data_filtrée_IM = data_filtrée_IM[data_filtrée_IM['lot_niveau'].isin(filt_niveau_IM)]
    if filt_lot_IM:
        data_filtrée_IM = data_filtrée_IM[data_filtrée_IM['lot_num'].isin(filt_lot_IM)]
    if filt_annee_IM:
        data_filtrée_IM = data_filtrée_IM[data_filtrée_IM['Annee'].isin(filt_annee_IM)]

    # Joindre CIQ et EEQ pour la même variable, Nickname (automate), année
    # Dans CIQ, on doit avoir colonne Année à créer (par exemple date d’analyse)
    # Ici on suppose CIQ a une colonne date, sinon on crée Année manuellement (à adapter)
    if 'Date' in data_filtrée_IM.columns:
        data_filtrée_IM['Date'] = pd.to_datetime(data_filtrée_IM['Date'], errors='coerce')
        data_filtrée_IM['Annee'] = data_filtrée_IM['Date'].dt.year
    else:
        st.warning("Pas de colonne Date dans CIQ : Année non disponible.")
        data_filtrée_IM['Annee'] = 0  # placeholder
    
    # st.dataframe(data_filtrée_IM)

    colonnes_valeurs_IM = data_filtrée_IM.columns[8:125]  
    
    data_filtrée_IM_long = data_filtrée_IM.melt(
        id_vars=["Nickname", "lot_niveau", "Annee","lot_num"],
        value_vars=colonnes_valeurs_IM,
        var_name="Paramètre",
        value_name="Valeur"
    )

    # 1. Convertir la colonne Valeur en numérique (force les erreurs en NaN)
    data_filtrée_IM_long['Valeur'] = pd.to_numeric(data_filtrée_IM_long['Valeur'], errors='coerce')

    # 2. Supprimer les lignes où la Valeur est NaN (très important pour le groupby)
    data_filtrée_IM_long = data_filtrée_IM_long.dropna(subset=['Valeur'])

    # st.dataframe(data_filtrée_IM_long.head())
    
    data_filtrée_IM_moyennes = (
    data_filtrée_IM_long.groupby(["Nickname", "Paramètre", "lot_niveau", "Annee"])
    .agg(moy_valeur=("Valeur", lambda x: pd.to_numeric(x, errors="coerce").mean()))
    .reset_index()
    )

    # st.dataframe(data_filtrée_IM_moyennes.head())
    
    EEQ["Resultat"] = (
        EEQ["Resultat"]
        .astype(str)  # au cas où il y aurait des nombres mélangés avec des strings
        .str.replace(",", ".", regex=False)
    )

    EEQ["Resultat"] = pd.to_numeric(EEQ["Resultat"], errors="coerce")
    data_filtrée_IM_moyennes["moy_valeur"] = pd.to_numeric(data_filtrée_IM_moyennes["moy_valeur"], errors="coerce")

    # Application sur ton DataFrame EEQ
    EEQ["lot_niveau_proche"] = EEQ.apply(lambda row: trouver_lot_niveau_proche(row, data_filtrée_IM_moyennes), axis=1)

    # st.dataframe(EEQ)

    biais_moyen = (
        EEQ.groupby(["Nickname", "Paramètre", "Annee", "lot_niveau_proche"])
        .agg(
            # Moyenne simple des biais (en valeur absolue si vous préférez)
            moy_biais=("Biais |c| pairs", lambda x: np.mean(np.abs(x.dropna()))),
            
            # Écart-type des biais
            sd_biais=("Biais |c| pairs", lambda x: np.std(x.dropna())),
            
            # Calcul de l'uBIAS (RMS=moy quadratique) : racine de la moyenne des carrés (Équivalent mathématique à sqrt(moyenne^2 + SD^2))
            uBIAS=("Biais |c| pairs", lambda x: np.sqrt(np.mean(x.dropna()**2)))
        )
        .reset_index()
    )
    


    # st.write("Aperçu des colonnes  :", biais_moyen.columns.tolist())
    # st.dataframe(biais_moyen.head())
    
    
    data_filtrée_IM_grouped = data_filtrée_IM_long.groupby(["Nickname", "lot_niveau", "Annee", "Paramètre"]).agg(
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

    # st.dataframe(data_filtrée_IM_grouped, hide_index = True)
    
    # Calcul spécifique du SD Pooled par groupe
    # Assurez-vous d'abord que 'lot_num' est bien présent dans votre DataFrame long
    if 'lot_num' not in data_filtrée_IM_long.columns:
        st.error("La colonne 'lot_num' est absente de data_filtrée_IM_long")
    else:
        # On calcule le SD Pooled
        # Note: On ne met PAS 'lot_num' dans le groupby principal
        sd_pooled_series = data_filtrée_IM_long.groupby(
            ["Nickname", "lot_niveau", "Annee", "Paramètre"], 
            group_keys=False
        ).apply(calcul_sd_pooled_custom, include_groups=False) 
        
        # On transforme la série en DataFrame pour le merge
        sd_pooled_df = sd_pooled_series.reset_index()
        sd_pooled_df.columns = ["Nickname", "lot_niveau", "Annee", "Paramètre", "SD_Pooled"]

        # Fusion avec votre tableau de stats existant
        data_filtrée_IM_grouped = data_filtrée_IM_grouped.merge(
            sd_pooled_df, 
            on=["Nickname", "lot_niveau", "Annee", "Paramètre"], 
            how="left"
        )
    
    # Calcul du SD Pooled Robuste par groupe (Nickname, Niveau, Année, Paramètre)
    sd_pooled_series = data_filtrée_IM_long.groupby(
        ["Nickname", "lot_niveau", "Annee", "Paramètre"], 
        group_keys=False
    ).apply(calcul_sd_pooled_robuste, include_groups=False)

    # 2. Préparation du DataFrame pour la fusion
    sd_pooled_df = sd_pooled_series.reset_index()
    sd_pooled_df.columns = ["Nickname", "lot_niveau", "Annee", "Paramètre", "SD_Pooled_Robuste"]

    # 3. Fusion avec votre tableau de synthèse final
    data_filtrée_IM_grouped = data_filtrée_IM_grouped.merge(
        sd_pooled_df, 
        on=["Nickname", "lot_niveau", "Annee", "Paramètre"], 
        how="left"
    )

    # df_IM = pd.merge(
    # biais_moyen,
    # CIQ_grouped_1102,
    # on=["Nickname", "Paramètre", "Annee"],
    # how="inner"  # ou "left", "right", "outer" selon ton besoin
    # )

    # Limites acceptables : sourve EFLM MAU min
    limites_en_pourcentage = {
    "WBC(10^9/L)": 16.7,
    "RBC(10^12/L)": 4.2,
    "HGB(g/L)": 4.1,
    "HCT(%)": 4.2,
    "PLT(10^9/L)": 11,
    "[PLT-F(10^9/L)]": 11,
    "RET#(10^9/L)": 14.6,
    "MCV(fL)": 1.2,
    "LYMPH#(10^9/L)": 15.8,
    "MONO#(10^9/L)": 21,
    "BASO#(10^9/L)": 18.9,
    "EO#(10^9/L)": 22.7,
    "NEUT#(10^9/L)": 18.8,
    "RET-He(pg)": 2.6,
    "MPV(fL)" : 3.5,
    "RDW-CV(%)" : 2.6,
    "MCH(pg)" : 1.1,
    "MCHC(g/dL)" : 1.5
    }
    
    # Limites acceptables : sourve PBQ U souhaitable
    limitesPBQ_en_pourcentage = {
    "WBC(10^9/L)": 14.8,
    "RBC(10^12/L)": 4.6,
    "HGB(g/L)": 4.3,
    "HCT(%)": 4.2,
    "PLT(10^9/L)": 11.4,
    "[PLT-F(10^9/L)]": 11.4,
    "MCV(fL)": 2.1,
    "MPV(fL)" : 4.4,
    "RDW-CV(%)" : 2.9,
    "MCH(pg)" : 2.5,
    "MCHC(g/dL)" : 1.3
    }

    df_IM = pd.merge(
        biais_moyen,
        data_filtrée_IM_grouped,
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

    def calculer_limite_absoluePBQ(row):
        param = row['Paramètre']
        moyenne = row['Moyenne']
        if param in limitesPBQ_en_pourcentage and pd.notnull(moyenne):
            return moyenne * limitesPBQ_en_pourcentage[param] / 100
        else:
            return np.nan

    df_IM['limite_accept'] = df_IM.apply(calculer_limite_absolue, axis=1)
    df_IM['limite_accept_PBQ'] = df_IM.apply(calculer_limite_absoluePBQ, axis=1)


    def highlight_status(row):
        # Initialisation d'une liste de styles vides (même taille que la ligne)
        styles = [''] * len(row)
        
        # Récupération dynamique des positions des colonnes
        u_idx = row.index.get_loc('U')
        statut_idx = row.index.get_loc('Statut')
        parametre_idx = row.index.get_loc('Paramètre')
        nickname_idx = row.index.get_loc('Nickname')
        
        # On vérifie si Statut_PBQ existe dans la ligne avant de chercher l'index
        pbq_idx = row.index.get_loc('Statut_PBQ') if 'Statut_PBQ' in row.index else None

        # Style à appliquer
        error_style = 'background-color: #FF4B4B; color: white; font-weight: bold'

        # Condition : Si l'un des deux statuts est Non Conforme
        # (Utilisation de .get pour éviter les erreurs si une colonne manque)
        is_nc_standard = row.get('Statut') == "❌ Non Conforme"
        is_nc_pbq = row.get('Statut_PBQ') == "❌ Non Conforme"

        if is_nc_standard or is_nc_pbq:
            styles[u_idx] = error_style
            styles[statut_idx] = error_style
            styles[parametre_idx] = error_style
            styles[nickname_idx] = error_style
            if pbq_idx is not None:
                styles[pbq_idx] = error_style
                
        return styles

        
    # st.dataframe(df_IM)
        
    # Calcul incertitudes
    # Colonnes proposées
    options_sd = ['SD_classique', 'SD_IQR', 'SD_IQR2', 'SD_MAD', 'SD_Pooled','SD_Pooled_Robuste']

    # Sélecteur unique
    choix_sd = st.selectbox("Choisissez le type de CV pour calculer u_CIQ :", options_sd, index = 5)

    # Affecter la colonne choisie à u_CIQ si elle existe dans df_IM
    if choix_sd in df_IM.columns:
        df_IM['u_CIQ'] = df_IM[choix_sd]
        st.write(f"Colonne `{choix_sd}` utilisée pour calculer 'u_CIQ'")
        # st.dataframe(df_IM[['u_CIQ']].head())
    else:
        st.warning(f"La colonne {choix_sd} n'existe pas dans les données.")
    
    
    df_IM['u_biais'] = df_IM['sd_biais']
    df_IM['u_total'] = np.sqrt(df_IM['u_biais']**2 + df_IM['u_CIQ']**2)
    df_IM['U'] = df_IM['u_total'] * 1.96  # élargie (k=1.96)
    df_IM['U%'] = 100 * df_IM['U'] / df_IM['Moyenne']

    # 1. Création de la colonne de conformité
    # On vérifie si U est supérieur à la limite_accept
    df_IM['Statut'] = np.where(
        df_IM['U'] > df_IM['limite_accept'], 
        "❌ Non Conforme", 
        "✅ Conforme"
    )

    df_IM['Statut_PBQ'] = np.where(
    df_IM['U'] > df_IM['limite_accept_PBQ'], 
    "❌ Non Conforme", 
    "✅ Conforme"
    )

    # Optionnel : Gérer les cas où les données sont manquantes (NaN)
    df_IM.loc[df_IM['U'].isnull() | df_IM['limite_accept'].isnull(), 'Statut'] = "Incomplet"
    df_IM.loc[df_IM['U'].isnull() | df_IM['limite_accept_PBQ'].isnull(), 'Statut_PBQ'] = "Incomplet"

    # st.dataframe(df_IM)
    
    # On récupère toutes les colonnes
    cols_df_IM = list(df_IM.columns)

    # On retire 'ID' et on le place au début
    cols_df_IM.insert(27, cols_df_IM.pop(cols_df_IM.index('limite_accept')))
    cols_df_IM.insert(28, cols_df_IM.pop(cols_df_IM.index('limite_accept_PBQ')))
    df_IM = df_IM[cols_df_IM]

    def style_gras(v):
        return 'font-weight: bold'

    # Application des styles
    styled_df = (
    df_IM.style
    .apply(highlight_status, axis=1)
    .map(style_gras, subset=['U','limite_accept', 'limite_accept_PBQ'])
    )

    # Affichage dans Streamlit
    st.dataframe(styled_df, width='stretch')

    
    with st.expander("🔬 Détails des calculs : SD Pooled et SD Pooled Robuste"):
        
        st.write("Méthodologie : Calcul du SD Pooled")

        # Affichage de la formule mathématique
        st.latex(r"""
        SD_{pooled} = \sqrt{\frac{\sum_{i=1}^{k} (n_i - 1) \cdot SD_i^2}{\sum_{i=1}^{k} n_i - k}}
        """)

        # Explication des termes pour votre dossier d'accréditation
        st.info(r"""
        **Légende :**
        - $SD_{pooled}$ : Écart-type combiné (Fidélité intermédiaire).
        - $k$ : Nombre de lots de contrôle différents sur la période.
        - $n_i$ : Nombre de mesures effectuées pour le lot $i$.
        - $SD_i$ : Écart-type calculé spécifiquement pour le lot $i$.
        """)
        st.write("""
        Cette méthode est recommandée pour estimer l'incertitude de mesure 
        sans surestimer la variance due aux changements de moyennes cibles entre les lots.
        """)
        
        st.write("Méthodologie : Calcul du SD Pooled Robuste")

        st.write("""
        Le **SD Pooled Robuste** estime l'imprécision du système en neutralisant les changements de lots 
        et les valeurs aberrantes. C'est l'indicateur privilégié pour la fidélité intermédiaire.
        """)
        
        # Formule LaTeX complète
        st.latex(r"""
        SD_{pooled\_rob} = \sqrt{\frac{\sum_{i=1}^{k} (n_i - 1) \cdot (MAD_i \cdot 1,4826)^2}{\sum_{i=1}^{k} n_i - k}}
        """)
        
        st.info(r"""
        **Composantes de la formule :**
        * $(MAD_i \cdot 1,4826)$ : Correspond au **SD robuste** du lot $i$. L'utilisation de la MAD (Median Absolute Deviation) permet d'ignorer les valeurs aberrantes.
        * $\sum (n_i - 1)$ : Somme des degrés de liberté de chaque lot.
        * $k$ : Nombre total de lots différents sur la période.
        """)

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
    st.plotly_chart(fig_IM, width='stretch')


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
    st.plotly_chart(fig_IM2, width='stretch')