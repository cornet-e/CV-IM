import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(page_title="Expertise CIQ Robuste", layout="wide")

st.title("📊 Analyse Comparative des CIQ (Multi-lots)")
st.markdown("Cette interface compare les méthodes de calcul de la précision inter-lots et illustre l'impact des approches robustes.")

# --- BARRE LATÉRALE ---
st.subheader("Source des Données")
col_input, _ = st.columns([1, 1])
with col_input:
    mode = st.radio("Sélectionnez le mode d'entrée :", 
                    ["Simulation de 3 lots", "Charger mon fichier (CSV/Excel)"], 
                    horizontal=True)
if mode == "Simulation de 3 lots":
    data_list = []
    # Configuration : Nom, Moyenne, SD, présence d'outliers
    configs = [("Lot 1", 100, 2, False), ("Lot 2", 104, 3, False), ("Lot 3", 98, 2, True)]
    for name, mu, sd, has_outliers in configs:
        values = np.random.normal(mu, sd, 500)
        if has_outliers:
            # Ajout d'outliers pour démontrer la robustesse
            values = np.concatenate([values, [mu+15, mu+18, mu-12]]) 
        data_list.append(pd.DataFrame({'Lot': name, 'Valeur': values}))
    df = pd.concat(data_list)
else:
    uploaded_file = st.sidebar.file_uploader("Upload (Excel/CSV)", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)
    else:
        st.info("Veuillez charger un fichier pour continuer.")
        st.stop()

# --- CALCUL DES STATISTIQUES PAR LOT ---
stats_list = []
for lot in df['Lot'].unique():
    subset = df[df['Lot'] == lot]['Valeur']
    n = len(subset)
    m = subset.mean()
    med = subset.median()
    sd_class = subset.std()
    
    # MAD normalisée (SD Robuste)
    mad_val = stats.median_abs_deviation(subset, scale='normal')
    
    # IQR normalisé (SD IQR)
    iqr_raw = stats.iqr(subset)
    sd_iqr = iqr_raw / 1.349
    
    stats_list.append({
        "Lot": lot,
        "N": n,
        "Moyenne": m,
        "Médiane": med,
        "SD Classique": sd_class,
        "CV Classique (%)": (sd_class/m)*100,
        "SD MAD": mad_val,
        "CV MAD (%)": (mad_val/med)*100,
        "SD IQR": sd_iqr,
        "CV IQR (%)": (sd_iqr/med)*100
    })

df_res = pd.DataFrame(stats_list)

# --- DONNEES BRUTES ---
donnees_brutes = df['Valeur']

# --- PRÉPARATION DES DONNÉES POOLÉES (Centrées-Réduites) ---
df_residus = []
moyenne_globale = df['Valeur'].mean()

for lot in df['Lot'].unique():
    subset = df[df['Lot'] == lot].copy()
    # On centre les données sur 0 et on les déplace à la moyenne globale
    subset['Valeur_Pool'] = (subset['Valeur'] - subset['Valeur'].mean()) + moyenne_globale
    df_residus.append(subset)

df_pool_plot = pd.concat(df_residus)

# --- CALCULS DES INDICATEURS POOLÉS ---
total_n = df_res['N'].sum()
k_lots = len(df_res)
df_total = total_n - k_lots

# 1. SD Poolé Robuste (Base MAD)
sum_sq_sd_mad = sum([(row['N']-1) * (row['SD MAD']**2) for idx, row in df_res.iterrows()])
sd_pooled_robust = np.sqrt(sum_sq_sd_mad / df_total)

# 2. CV Poolé DÉRIVÉ du SD Poolé
# Formule : (SD Poolé / Moyenne des moyennes) * 100
avg_mean = df_res['Moyenne'].mean()
cv_pooled_derived = (sd_pooled_robust / avg_mean) * 100

# 3. CV Poolé DIRECT (Moyenne quadratique des CV MAD)
# Formule : sqrt( sum((n-1)*CV^2) / sum(n-1) )
sum_sq_cv_mad = sum([(row['N']-1) * (row['CV MAD (%)']**2) for idx, row in df_res.iterrows()])
cv_pooled_direct = np.sqrt(sum_sq_cv_mad / df_total)

# --- CALCUL DU CV ROBUSTE GLOBAL (TOUT MÉLANGÉ) ---
toutes_valeurs = df['Valeur']
moyenne_globale = toutes_valeurs.mean()
mediane_globale = toutes_valeurs.median()

# SD MAD Global
mad_global = stats.median_abs_deviation(toutes_valeurs, scale='normal')
cv_mad_global = (mad_global / mediane_globale) * 100

# SD IQR Global
iqr_global = stats.iqr(toutes_valeurs) / 1.349
cv_iqr_global = (iqr_global / mediane_globale) * 100

# CV Classique Global (pour comparaison)
cv_classique_global = (toutes_valeurs.std() / moyenne_globale) * 100


# --- AFFICHAGE GRAPHIQUE ---
st.subheader("1.1. Distributions des Valeurs")
fig, ax = plt.subplots(figsize=(12, 5))
sns.kdeplot(data=df, x="Valeur", hue="Lot", fill=True, alpha=0.3, palette="viridis", ax=ax)
st.pyplot(fig)

st.subheader("1.2. Distributions des Valeurs et Courbe Poolée")
fig, ax = plt.subplots(figsize=(12, 6))

# Courbes individuelles par lot
sns.kdeplot(data=df, x="Valeur", hue="Lot", fill=True, alpha=0.2, palette="viridis", ax=ax)

# Courbe Poolée Globale (en pointillés noirs)
sns.kdeplot(data=df_pool_plot, x="Valeur_Pool", color="black", linestyle="--", 
            linewidth=2.5, label="Distribution Poolée (Référence)", ax=ax)

ax.set_title("Comparaison des lots vs Distribution Poolée Globale")
ax.legend()
st.pyplot(fig)

st.subheader("1.3. Comparaison : Lots Individuels, Poolé vs Mélange Brut")
fig, ax = plt.subplots(figsize=(12, 6))

# Courbes individuelles (fines)
sns.kdeplot(data=df, x="Valeur", hue="Lot", fill=True, alpha=0.1, palette="viridis", ax=ax, linewidth=1)

# Courbe Poolée (Pointillés noirs) - Représente la précision réelle
sns.kdeplot(df_pool_plot['Valeur_Pool'], color="black", linestyle="--", 
            linewidth=2.5, label="Distribution Poolée (Précision réelle)", ax=ax)

# Courbe Brute (Ligne pleine rouge) - Le mélange de tous les points
sns.kdeplot(donnees_brutes, color="red", linestyle="-", 
            linewidth=2, label="Mélange Brut (Effet Biais + Précision)", ax=ax)

ax.set_title("Impact du mélange des lots sur la distribution")
ax.legend()
st.pyplot(fig)

# --- AFFICHAGE TABLEAU DES PERFORMANCES ---
st.subheader("2. Indicateurs de Performance par Lot")
st.dataframe(df_res.style.format(precision=3).highlight_max(subset=['SD Classique'], color='#ffcccc'))

# --- RÉSULTATS POOLÉS ---
st.subheader("3. Synthèse de la Performance Globale (Poolée)")
c1, c2, c3, c4 = st.columns(4)

c1.metric("SD Poolé Robuste", f"{sd_pooled_robust:.3f}", help="Dispersion absolue moyenne basée sur la MAD")
c2.metric("CV Poolé Dérivé (%)", f"{cv_pooled_derived:.2f} %", help="Calculé via : (SD Poolé / Moyenne Globale)")
c3.metric("CV Poolé Direct (%)", f"{cv_pooled_direct:.2f} %", help="Moyenne quadratique des CV robustes de chaque lot")
c4.metric("CV Robuste Global (%)", f"{cv_mad_global:.2f} %", help="Calculé sur TOUTES les données sans distinction de lot")

st.info("💡 Note : Le **CV Poolé Direct** est généralement privilégié en biologie car il reflète la précision relative indépendamment du niveau de concentration.")

# --- FORMULES MATHÉMATIQUES ---
with st.expander("📚 MÉTHODOLOGIE ET FORMULES MATHÉMATIQUES"):
    st.markdown(r"""
    ### 1. Estimations Robustes (Normalisées)
    * **SD MAD :** $$SD_{MAD} = 1.4826 \times \text{médiane}(|x_i - \tilde{x}|)$$
    * **SD IQR :** $$SD_{IQR} = \frac{Q3 - Q1}{1.349}$$

    ### 2. Calculs Poolés (Multi-lots)
    Le pooling pondère la dispersion par les degrés de liberté ($n-1$) de chaque lot.

    * **SD Poolé Robuste :** $$SD_{poolé} = \sqrt{\frac{\sum (n_i - 1) \cdot SD_{MAD,i}^2}{\sum (n_i - 1)}}$$

    * **CV Poolé Dérivé :** Calculé à partir du SD poolé global.  
      $$CV_{poolé\_der} = \frac{SD_{poolé}}{\bar{X}_{globale}} \times 100$$

    * **CV Poolé Direct (Recommandé) :** Moyenne quadratique des CV robustes.  
      $$CV_{poolé\_dir} = \sqrt{\frac{\sum (n_i - 1) \cdot CV_{MAD,i}^2}{\sum (n_i - 1)}}$$
    
    * **CV Robuste Global (Mélange Brut) :** Ce calcul traite l'ensemble des données comme un seul échantillon géant.  
      $$CV_{global\_rob} = \frac{1.4826 \times \text{MAD}(\text{toutes données})}{\text{médiane globale}} \times 100$$  
      Contrairement au **CV Poolé**, cette mesure est influencée par l'écart entre les moyennes des lots (le biais inter-lot). Elle reste cependant "robuste" face aux erreurs analytiques isolées.
    """)