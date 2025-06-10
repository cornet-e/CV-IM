import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from io import StringIO
from datetime import datetime


# === Fonctions de calcul de CV ===
def cv(x):
    x = pd.to_numeric(x, errors='coerce')
    m = np.nanmean(x)
    return np.nan if m == 0 or np.isnan(m) else (np.nanstd(x) / m) * 100

def cv_robuste_iqr(x):
    x = pd.to_numeric(x, errors='coerce')
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    return np.nan if med == 0 or np.isnan(med) else (iqr / med) * 100

def cv_robuste_iqr2(x):
    x = pd.to_numeric(x, errors='coerce')
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    sigma_robuste = iqr / 1.349
    return np.nan if med == 0 or np.isnan(med) else (sigma_robuste / med) * 100

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






# === Chargement des fichiers ===
uploaded_files = st.file_uploader("Importer un ou plusieurs fichiers CSV", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    list_df = []
    for file in uploaded_files:
        try:
            content = file.read().decode('utf-8', errors='replace').splitlines()
        except UnicodeDecodeError:
            file.seek(0)
            content = file.read().decode('cp1252', errors='replace').splitlines()

        if len(content) < 2:
            st.warning(f"Le fichier {file.name} semble vide ou mal formaté.")
            continue

        lines = content[1:]  # on ignore la 1ère ligne
        content_str = StringIO('\n'.join(lines))

        try:
            df = pd.read_csv(content_str, sep=',', on_bad_lines='skip')
        except Exception as e:
            st.error(f"Erreur lecture du fichier {file.name}: {e}")
            continue

        list_df.append(df)

    if list_df:
        CIQ = pd.concat(list_df, ignore_index=True)
        st.success(f"{len(uploaded_files)} fichier(s) chargé(s), total : {CIQ.shape[0]} lignes.")
        st.dataframe(CIQ.head())
    else:
        st.warning("Aucun fichier n'a pu être chargé correctement.")
else:
    st.stop()

# Modifier l'unité de l'Hb : g/dL => g/L
CIQ['HGB(g/dL)'] = pd.to_numeric(CIQ['HGB(g/dL)'], errors='coerce') * 10
# Renommer la colonne
CIQ.rename(columns={'HGB(g/dL)': 'HGB(g/L)'}, inplace=True)
st.dataframe(CIQ.head())

# === Chargement de la liste des champs ===
try:
    liste_champs_df = pd.read_csv("liste_champs.csv", sep=',', encoding="utf-8")
except UnicodeDecodeError:
    liste_champs_df = pd.read_csv("liste_champs.csv", sep=',', encoding="cp1252")

if liste_champs_df.shape[1] == 1:
    colonnes_voulues = liste_champs_df.iloc[:, 0].dropna().astype(str).str.strip()
else:
    colonnes_voulues = liste_champs_df.columns.astype(str).str.strip()

colonnes_finales = [col for col in colonnes_voulues if col in CIQ.columns]
if not colonnes_finales:
    st.error("Aucune des colonnes attendues n'a été trouvée dans le fichier CIQ.")
    st.stop()

CIQ = CIQ[colonnes_finales]
st.success(f"{len(colonnes_finales)} colonnes conservées dans le fichier CIQ.")



# === Détection automatique des colonnes automate et lot ===
colonnes_automate = [col for col in CIQ.columns if 'nick' in col.lower()]
colonnes_lot = [col for col in CIQ.columns if 'sample' in col.lower() and 'no' in col.lower()]

if not colonnes_automate or not colonnes_lot:
    st.error("Colonnes 'automate' ou 'lot' non trouvées automatiquement.")
    st.write("Colonnes disponibles :", CIQ.columns.tolist())
    st.stop()

col_automate = st.selectbox("Colonne automate :", colonnes_automate, key="automate")
col_lot = st.selectbox("Colonne lot (sample no) :", colonnes_lot, key="lot")
CIQ[col_automate] = CIQ[col_automate].astype(str)


# === Création des colonnes lot_num et lot_niveau ===
CIQ['lot_num'] = CIQ[col_lot].astype(str).str[:18]
CIQ['lot_niveau'] = CIQ[col_lot].astype(str).str[18:22]
# Extraire Année
CIQ['Date'] = pd.to_datetime(CIQ['Date'], errors='coerce')
CIQ['Annee'] = CIQ['Date'].dt.year.astype("Int64")



st.subheader("Graph par paramètre sélectionné (détail par année)")

# === Choix du paramètre ===
choix_param = CIQ.columns[8:]  # adapter si besoin
param = st.selectbox("Choisissez le paramètre à étudier", choix_param)

# Sélection dynamique des critères

filt_automate = st.multiselect("Automate(s)", sorted(CIQ[col_automate].dropna().unique()), default=None)
filt_niveau = st.multiselect("Niveau(x) de lot", sorted(CIQ['lot_niveau'].dropna().unique()), default=['1101', '1102', '1103'])
filt_annee = st.multiselect("Année(s)", sorted(CIQ['Annee'].dropna().unique()), default=None)


# Filtrage des données
data_filtrée = CIQ.copy()
if filt_automate:
    data_filtrée = data_filtrée[data_filtrée[col_automate].isin(filt_automate)]
if filt_niveau:
    data_filtrée = data_filtrée[data_filtrée['lot_niveau'].isin(filt_niveau)]
if filt_annee:
    data_filtrée = data_filtrée[data_filtrée['Annee'].isin(filt_annee)]

# Conversion du paramètre sélectionné en float
data_filtrée[param] = pd.to_numeric(data_filtrée[param], errors='coerce')


# st.dataframe(data_filtrée)

# Agrégation par automate et niveau
grouped = data_filtrée.groupby([col_automate, 'lot_niveau','Annee'])[param].agg(
    n='count',
    Moyenne='mean',
    Ecart_type='std',
    CV=cv,
    CV_IQR=cv_robuste_iqr,
    CV_IQR2=cv_robuste_iqr2,
    CV_MAD=cv_robuste_mad
).reset_index()

st.dataframe(grouped)

grouped['lot_annee'] = grouped['lot_niveau'].astype(str) + " (" + grouped['Annee'].astype(str) + ")"


# Graphs interactifs
#def plot_cv(y, title, ylabel):
#    fig = px.bar(grouped, x='lot_niveau', y=y, color=col_automate,
#                 barmode='group',
#                 hover_data=['n'],
#                 title=title,
#                 labels={y: ylabel, 'lot_niveau': 'Niveau de lot'})
#    st.plotly_chart(fig)

def plot_cv(y, title, ylabel):
    fig = px.bar(grouped, x='lot_annee', y=y, color=col_automate,
                 barmode='group',
                 hover_data=['n', 'Annee', 'lot_niveau'],
                 title=title,
                 labels={y: ylabel, 'lot_annee': 'Niveau de lot (Année)'}
                )
    st.plotly_chart(fig)


st.subheader("Graphiques des CV")
plot_cv("CV", f"{param} : CV classique", "CV (%)")
plot_cv("CV_IQR", f"{param} : CV IQR", "CV (%)")
plot_cv("CV_IQR2", f"{param} : CV IQR robuste", "CV (%)")
plot_cv("CV_MAD", f"{param} : CV MAD", "CV (%)")

# =======================
# ➕ Graphique Facets (CV_MAD par paramètre)
# =======================

st.subheader("Facets : CV MAD par paramètre (moyenne de tous les CIQ)")


# Sélectionne les colonnes de l'index 8 à 125 pour permettre la conversion en numérique
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
    'WBC(10^3/uL)','RBC(10^6/uL)','HGB(g/dL)','HCT(%)','MCV(fL)','MCH(pg)','MCHC(g/dL)','PLT(10^3/uL)','[RBC-O(10^6/uL)]','[PLT-O(10^3/uL)]','[PLT-F(10^3/uL)]','IPF#(10^3/uL)','[HGB-O(g/dL)]'
    ]

# Sélecteur des paramètres à inclure dans les facets
params_selectionnés = st.multiselect(
    "Paramètres à afficher en facets",
    options=params_numeriques,
    default=[p for p in params_visibles_par_défaut if p in params_numeriques]
)


# Compilation des CV_MAD pour chaque paramètre
liste_dfs = []
for p in params_selectionnés:
    df_tmp = data_filtrée.groupby([col_automate, 'lot_niveau'])[p].agg(
        n='count',
        CV_MAD=cv_robuste_mad
    ).reset_index()
    df_tmp['paramètre'] = p
    df_tmp.rename(columns={'CV_MAD': 'CV'}, inplace=True)
    liste_dfs.append(df_tmp)



# Fusion de tous les DataFrames
df_facet = pd.concat(liste_dfs, ignore_index=True)


# Graphique facets avec Plotly Express
fig_facet = px.bar(
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

fig_facet.update_yaxes(matches=None)  # axes Y indépendants


# Affiche tous les labels d’axe Y
for axis in fig_facet.layout:
    if axis.startswith("yaxis"):
        fig_facet.layout[axis].showticklabels = True
        fig_facet.layout[axis].title = dict(text="CV MAD (%)")


# Forcer l’affichage de l’axe X et du titre sur chaque subplot
for axis_name in fig_facet.layout:
    if axis_name.startswith("xaxis"):
        axis = fig_facet.layout[axis_name]
        axis.showticklabels = True  # Affiche les ticks
        axis.title = dict(text="Niveau de lot")  # Titre de l'axe X


fig_facet.update_layout(height=300 * ((len(params_selectionnés) - 1) // 3 + 1))  # ajuste la hauteur automatiquement
st.plotly_chart(fig_facet)




# =======================
# ➕ Graphique Facets ( valeur paramètre) avec filtre année
# =======================

st.subheader("Facets : Distribution des valeurs de chaque paramètre")

    
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

# --- Chargement fichiers ---
st.title("Incertitudes élargies")

uploaded_eeq = st.file_uploader("Importer fichier EEQ (exportEEQ.csv)", type=["csv"])

if uploaded_eeq:
    EEQ = pd.read_csv(uploaded_eeq, sep=";", encoding="ISO-8859-1", on_bad_lines="skip")
   
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
        return np.nan

    EEQ['Nickname'] = EEQ.apply(assign_nickname, axis=1)

    # Ajouter variable selon Analyte (mapping R -> Python)
    analyte_map = {
        "CCMH": "MCHC(g/dL)",
        "Frac Plaq immatures (IPF)": "IPF(%)",
        "Frac Réti immatures (IRF)": "IRF(%)",
        "Hématies (impédance)": "RBC(10^6/uL)",
        "Hématies (optique)": "[RBC-O(10^6/uL)]",
        "Hématocrite": "HCT(%)",
        "Hémoglobine": "HGB(g/L)",
        "IDR": "RDW-CV(%)",
        "Leucocytes": "WBC(10^3/uL)",
        "Lymphocytes": "LYMPH%(%)",
        "Lymphocytes (abs)": "LYMPH#(10^3/uL)",
        "Monocytes": "MONO%(%)",
        "Monocytes (abs)": "MONO#(10^3/uL)",
        "Plaquettes (fluorescence)": "[PLT-F(10^3/uL)]",
        "Plaquettes (impédance)": "PLT(10^3/uL)",
        "Plaquettes (optique)": "[PLT-O(10^3/uL)]",
        "Poly. Basophiles": "BASO%(%)",
        "Poly. Basophiles (abs)": "BASO#(10^3/uL)",
        "Poly. Eosinophiles": "EO%(%)",
        "Poly. Eosinophiles (abs)": "EO#(10^3/uL)",
        "Poly. Neutrophiles": "NEUT%(%)",
        "Poly. Neutrophiles (abs)": "NEUT#(10^3/uL)",
        "R-MFV (Volume Erythro. le plus fréquent)": "R-MFV(fL)"
    }
    EEQ['variable'] = EEQ['Analyte'].map(analyte_map)

    # Extraire Année
    EEQ['Annee'] = EEQ['Date'].dt.year
    # EEQ['HGB(g/dL)'] = EEQ['HGB(g/dL)'] / 10 # conversion d'unité de g/L à g/dL
    
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
    st.dataframe(EEQ)
 



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
    "WBC(10^3/uL)": 15.49,
    "RBC(10^6/uL)": 4.4,
    "HGB(g/dL)": 4.19,
    "HCT(%)": 3.97,
    "PLT(10^3/uL)": 13.4,
    "[PLT-F(10^3/uL)]": 13.4,
    "RET#(10^3/uL)": 16.8,
    "VGM(fL)": 2.42,
    "LYMPH#(10^3/uL)": 17.6,
    "MONO#(10^3/uL)": 27.9,
    "BASO#(10^3/uL)": 38.5,
    "EO#(10^3/uL)": 37.1,
    "NEUT#(10^3/uL)": 23.35
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
    df_IM['u_biais'] = df_IM['sd_biais']
    df_IM['u_CIQ'] = df_IM['SD_MAD']
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
        'WBC(10^3/uL)','RBC(10^6/uL)','HGB(g/dL)','HCT(%)','MCV(fL)','MCH(pg)','MCHC(g/dL)','PLT(10^3/uL)','[RBC-O(10^6/uL)]','[PLT-O(10^3/uL)]','[PLT-F(10^3/uL)]','IPF#(10^3/uL)','[HGB-O(g/dL)]'
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

    fig_IM = px.bar(
        df_IM_filtré,
        x="Annee",
        y="U",
        color="Nickname",
        facet_row="lot_niveau_proche",     # ➜ 1 ligne par lot_niveau_proche
        facet_col="Paramètre",             # ➜ 1 colonne par paramètre
        facet_row_spacing=0.1,
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
    
    fig_IM.update_layout(
    height=max(300, 250 * nb_lots * nb_params),  # Ajustement plus fin si tu veux
)

import plotly.graph_objects as go
# 📌 Récupérer l'ordre des facettes **tel que Plotly Express les affiche**
facet_row_order = sorted(df_IM_filtré["lot_niveau_proche"].unique(), key=lambda x: str(x))
facet_col_order = sorted(df_IM_filtré["Paramètre"].unique(), key=lambda x: str(x))

st.write("Ordre des facettes - lot_niveau_proche:", facet_row_order)
st.write("Ordre des facettes - Paramètre:", facet_col_order)

# Ajout des points pour 'limite_accept' en respectant l'affichage des facettes
for lot in facet_row_order:
    for param in facet_col_order:
        df_subset = df_IM_filtré[(df_IM_filtré["lot_niveau_proche"] == lot) & (df_IM_filtré["Paramètre"] == param)]
        
        fig_IM.add_trace(
            go.Scatter(
                x=df_subset["Annee"],
                y=df_subset["limite_accept"],
                mode="markers",
                marker=dict(color="red", size=8),
                name=f"Limite acceptée - {lot}, {param}",
            ),
            row=facet_row_order.index(lot) + 1,  # 🔥 Utilisation correcte des indices réels
            col=facet_col_order.index(param) + 1  # 🔥 Alignement parfait des colonnes
        )


    # fig_IM.update_layout(height=300 * len(param_selectionnes))
    st.plotly_chart(fig_IM, use_container_width=True)



else:
    st.info("Importer le fichier EEQ pour démarrer.")
