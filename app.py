import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import numpy as np
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Diag Flash Données Sociales", layout="wide")

# ==========================================
# 🔒 SÉCURITÉ
# ==========================================
MOT_DE_PASSE = "ericpeltier"
LOGO_FILE = "logoE5.png" 

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    if st.session_state.password_input == MOT_DE_PASSE:
        st.session_state.authenticated = True
        del st.session_state.password_input
    else:
        st.error("Mot de passe incorrect ❌")

if not st.session_state.authenticated:
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=200)
    else:
        st.warning(f"Image '{LOGO_FILE}' introuvable.")
        
    st.title("🔒 Accès Restreint")
    st.markdown("### Outil Diag Flash Données Sociales")
    st.text_input("Veuillez saisir le mot de passe :", type="password", key="password_input", on_change=check_password)
    st.stop()

# ==========================================
# 📂 CHARGEMENT ROBUSTE & EXPORT
# ==========================================

def smart_header_detection(df):
    expected_keywords = {'MATRICULE', 'MAT', 'SEXE', 'GENRE', 'SORTIE', 'ENTREE', 'SERVICE', 'EMPLOI'}
    current_cols = set(str(c).upper().strip() for c in df.columns)
    if len(current_cols.intersection(expected_keywords)) >= 2:
        return df
    for i, row in df.head(5).iterrows():
        row_values = set(str(val).upper().strip() for val in row.values if pd.notna(val))
        if len(row_values.intersection(expected_keywords)) >= 2:
            new_header = row
            df_new = df.iloc[i+1:].copy()
            df_new.columns = new_header
            return df_new
    return df

@st.cache_data
def load_data(file):
    df = None
    try: df = pd.read_excel(file)
    except: pass
    if df is None:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=';', encoding='latin-1', on_bad_lines='skip')
            if len(df.columns) < 2:
                file.seek(0)
                df = pd.read_csv(file, sep=',', encoding='latin-1', on_bad_lines='skip')
        except: pass
    if df is None:
        try:
            file.seek(0)
            df = pd.read_csv(file, sep=',', encoding='utf-8', on_bad_lines='skip')
        except: pass
    if df is not None:
        df = smart_header_detection(df)
        df.columns = df.columns.astype(str).str.strip().str.upper()
        return df
    return None

def extract_year(filename):
    match = re.search(r'20\d{2}', filename)
    return int(match.group(0)) if match else None

def convert_df(df):
    return df.to_csv(index=False, sep=';', encoding='utf-8-sig')

# PALETTE COULEURS GENERALE
extended_palette = (
    px.colors.qualitative.G10 + 
    px.colors.qualitative.T10 + 
    px.colors.qualitative.Bold + 
    px.colors.qualitative.Vivid + 
    px.colors.qualitative.Alphabet + 
    px.colors.qualitative.Dark24
)

COLOR_MAP_SEXE = {
    'H': '#87CEEB', 'M': '#87CEEB', 'HOMME': '#87CEEB', 'HOMMES': '#87CEEB',
    'F': '#00008B', 'FEMME': '#00008B', 'FEMMES': '#00008B'
}

def get_gender_color(val):
    val_str = str(val).upper().strip()
    if val_str.startswith('H') or val_str.startswith('M'): return '#87CEEB' 
    elif val_str.startswith('F'): return '#00008B' 
    return '#808080' 

# ==========================================
# 🚀 NAVIGATION & CHARGEMENT (SIDEBAR)
# ==========================================
if os.path.exists(LOGO_FILE):
    st.sidebar.image(LOGO_FILE, width=180)

st.sidebar.title("Navigation")

menu_options = [
    "🔺 Pyramides & Ratios",
    "📍 Cartographie Structures",
    "🧿 Micro-Analyse",
    "🚪 Analyse des Sorties (Turn-over)",
    "📈 Évolution Effectifs",
    "📋 Types de Contrat",
    "📉 Flux (Histo Décalé)",
    "📊 Absentéisme (Évolution)",
    "⚖️ Absentéisme (Comparaison)",
    "💰 Autres (Promo/Form/Rém)",
    "📝 Restrictions Santé",
    "📚 Ressources & Guides"
]

selection_page = st.sidebar.radio("Aller vers :", menu_options)

st.sidebar.markdown("---")

# On cache les uploaders sur la page Ressources pour simplifier l'interface
if selection_page != "📚 Ressources & Guides":
    st.sidebar.header("1. Données Effectifs (Stock)")
    uploaded_files = st.sidebar.file_uploader("Chargez vos fichiers annuels", accept_multiple_files=True)

    st.sidebar.header("2. Données Sorties (Flux)")
    uploaded_file_sorties = st.sidebar.file_uploader("Fichier 'Sorties' unique (Optionnel)", accept_multiple_files=False)

    min_eff_global = st.sidebar.slider("Taille min. groupes", 1, 20, 3)
else:
    uploaded_files = []
    uploaded_file_sorties = None
    min_eff_global = 3

# --- TRAITEMENT FICHIERS STOCK ---
data_dict = {}
combined_df = pd.DataFrame()

if uploaded_files:
    temp_dfs = []
    for f in uploaded_files:
        year = extract_year(f.name)
        if year:
            df = load_data(f)
            if df is not None:
                rename_map = {
                    'SERVICE / SECTEUR': 'SERVICE', 'SECTEUR': 'SERVICE', 'UNITÉ': 'SERVICE', 'AGENCE': 'SERVICE',
                    'DURÉE DU TRAVAIL': 'TEMPS_TRAVAIL',
                    'ARRIVÉE': 'ENTREE', 'DATE ENTREE': 'ENTREE', 'DATE D\'ENTRÉE': 'ENTREE',
                    'NAISSANCE': 'NAISSANCE', 'DATE NAISSANCE': 'NAISSANCE',
                    'TYPE CONTRAT': 'CONTRAT', 'NATURE CONTRAT': 'CONTRAT', 'STATUT': 'CONTRAT',
                    'RESTRICTIONS': 'RESTRICTION', 'RESTRICTION': 'RESTRICTION', 'AVIS': 'RESTRICTION', 
                    'COMMENTAIRE': 'RESTRICTION', 'APTITUDE': 'RESTRICTION'
                }
                cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
                df = df.rename(columns=cols_to_rename)
                df['ANNEE_FICH'] = year
                if 'NAISSANCE' in df.columns:
                    df['Date_Naiss'] = pd.to_datetime(df['NAISSANCE'], dayfirst=True, errors='coerce')
                    df['AGE_CALC'] = year - df['Date_Naiss'].dt.year
                    df.loc[(df['AGE_CALC'] < 14) | (df['AGE_CALC'] > 80), 'AGE_CALC'] = None
                if 'ENTREE' in df.columns:
                    df['Date_Entree'] = pd.to_datetime(df['ENTREE'], dayfirst=True, errors='coerce')
                    df['ANC_CALC'] = year - df['Date_Entree'].dt.year
                    df.loc[(df['ANC_CALC'] < 0) | (df['ANC_CALC'] > 60), 'ANC_CALC'] = None
                data_dict[str(year)] = df
                temp_dfs.append(df)
    if temp_dfs:
        combined_df = pd.concat(temp_dfs, ignore_index=True)
        sorted_years = sorted(data_dict.keys())
        st.sidebar.success(f"✅ Stock : {len(data_dict)} années")
        cat_cols = [c for c in ['SERVICE', 'EMPLOI', 'SEXE', 'CATEGORIE', 'POSTE'] if c in combined_df.columns]
    else: sorted_years = []

# --- TRAITEMENT FICHIER SORTIES ---
df_sorties = pd.DataFrame()
if uploaded_file_sorties:
    df_s = load_data(uploaded_file_sorties)
    if df_s is not None:
        rename_map_s = {
            'SERVICE / SECTEUR': 'SERVICE', 'SECTEUR': 'SERVICE', 'EMPLOI': 'EMPLOI', 'METIER': 'EMPLOI', 'POSTE': 'EMPLOI',
            'GENRE': 'SEXE', 'DATE ENTREE': 'ENTREE', 'ENTRÉE': 'ENTREE', 'DATE NAISSANCE': 'NAISSANCE',
            'SORTIE': 'SORTIE', 'SORTIES': 'SORTIE', 'DATE SORTIE': 'SORTIE', 'DATE DE SORTIE': 'SORTIE', 'FIN DE CONTRAT': 'SORTIE',
            'DATE_SORTIE': 'SORTIE', 'D_SORTIE': 'SORTIE', 'MOTIF': 'MOTIF', 'RAISON': 'MOTIF', 'TYPE CONTRAT': 'CONTRAT', 'CONTRAT': 'CONTRAT'
        }
        cols_to_rename_s = {k: v for k, v in rename_map_s.items() if k in df_s.columns}
        df_s = df_s.rename(columns=cols_to_rename_s)
        if 'SORTIE' not in df_s.columns:
            for col in df_s.columns:
                if "SORTIE" in col or "DEPART" in col or "FIN CONTRAT" in col:
                    df_s = df_s.rename(columns={col: 'SORTIE'}); break
        if 'SORTIE' in df_s.columns:
            df_s['Date_Sortie'] = pd.to_datetime(df_s['SORTIE'], dayfirst=True, errors='coerce')
            df_s['ANNEE_SORTIE'] = df_s['Date_Sortie'].dt.year
            if 'NAISSANCE' in df_s.columns:
                df_s['Date_Naiss'] = pd.to_datetime(df_s['NAISSANCE'], dayfirst=True, errors='coerce')
                df_s['AGE_SORTIE'] = (df_s['Date_Sortie'] - df_s['Date_Naiss']).dt.days / 365.25
                df_s['TRANCHE_AGE'] = pd.cut(df_s['AGE_SORTIE'], bins=[0,25,35,45,55,100], labels=['<25 ans','25-35 ans','35-45 ans','45-55 ans','55+ ans'])
            if 'ENTREE' in df_s.columns:
                df_s['Date_Entree'] = pd.to_datetime(df_s['ENTREE'], dayfirst=True, errors='coerce')
                df_s['ANC_SORTIE'] = (df_s['Date_Sortie'] - df_s['Date_Entree']).dt.days / 365.25
                df_s['TRANCHE_ANC'] = pd.cut(df_s['ANC_SORTIE'], bins=[0,1,3,5,10,100], labels=['<1 an','1-3 ans','3-5 ans','5-10 ans','10+ ans'])
        df_sorties = df_s
        st.sidebar.success(f"✅ Sorties : {len(df_sorties)} lignes")

st.title("Diag Flash Données Sociales_Application conçue par Eric PELTIER")

# ==========================================
# 📄 LOGIQUE DES PAGES
# ==========================================

# --- PAGE SPECIALE : RESSOURCES & GUIDES (NOUVEAU) ---
if selection_page == "📚 Ressources & Guides":
    st.header("📚 Bibliothèque de Ressources & Guides")
    st.markdown("Accédez aux fiches pratiques (PDF) stockées localement et aux sites de référence.")
    
    tab1, tab2 = st.tabs(["📄 Fiches Pratiques (PDF)", "🔗 Liens Utiles & Ouvrages"])
    
    with tab1:
        st.subheader("Documents à télécharger")
        # Détection AUTOMATIQUE des PDF dans le dossier 'ressources'
        folder_path = "ressources"
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            if files:
                st.success(f"{len(files)} document(s) trouvé(s) dans le dossier 'ressources'.")
                for file_name in sorted(files):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "rb") as f:
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.write(f"📄 **{file_name}**")
                        with c2:
                            st.download_button(
                                label="📥 Télécharger",
                                data=f,
                                file_name=file_name,
                                mime="application/pdf",
                                key=file_name
                            )
                    st.divider()
            else:
                st.info("📂 Le dossier 'ressources' est vide. Ajoutez vos fichiers PDF dedans pour qu'ils apparaissent ici.")
        else:
            st.warning("⚠️ Dossier 'ressources' introuvable. Veuillez créer un dossier nommé 'ressources' à côté de l'application.")

    with tab2:
        st.subheader("Sites Institutionnels & Ouvrages")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🌐 Sites Web de Référence")
            st.markdown("Des ressources pour approfondir les thématiques du travail.")
            
            st.link_button("ANACT : Travail des Seniors", "https://www.anact.fr/travail-seniors", use_container_width=True)
            st.link_button("GIS CREAPT (Cnam) : Expérience & Âge", "https://ceet.cnam.fr/partenariats/gis-creapt/centre-de-recherches-sur-l-experience-l-age-et-les-populations-au-travail-creapt--959490.kjsp", use_container_width=True)

        with c2:
            st.markdown("#### 📖 Bibliographie")
            st.markdown("**« La démographie du travail pour anticiper le vieillissement »**")
            st.caption("De Serge Volkoff et Anne-Françoise Molinié")
            st.markdown("Un ouvrage de référence sur l'ergonomie et la démographie du travail.")
            
            st.link_button("🔎 Voir la fiche de lecture (Anact)", "https://veille-travail.anact.fr/osiros/result/notice.php?queryosiros=id:31342&referer=home", use_container_width=True)

# --- BLOC D'ANALYSE CLASSIQUE (Si fichiers chargés) ---
elif not combined_df.empty or not df_sorties.empty:
    
    if selection_page == "🔺 Pyramides & Ratios":
        if combined_df.empty: st.warning("Veuillez charger des fichiers Effectifs (Stock)."); st.stop()
        st.header("Analyse Structurelle : Pyramides et Indicateurs Clés")
        
        c_yr, c_var = st.columns(2)
        yr_pyr = c_yr.selectbox("Année", sorted_years, index=len(sorted_years)-1, key="pyr_year")
        var_pyr = c_var.selectbox("Variable", ["Âge", "Ancienneté"], key="pyr_var")
        
        df_pyr = data_dict[str(yr_pyr)].copy()
        st.markdown("---")
        
        c_filt, c_subfilt = st.columns(2)
        opts_pyr_filter = ["Global (Toute l'entreprise)"] + cat_cols
        mode_filtre = c_filt.selectbox("Périmètre", opts_pyr_filter, key="pyr_filtre")
        
        subset_df = df_pyr.copy()
        titre_complement = "Global"
        if mode_filtre != "Global (Toute l'entreprise)":
            valeurs_possibles = sorted(subset_df[mode_filtre].dropna().unique())
            val_retenue = c_subfilt.selectbox(f"Choisir {mode_filtre}", valeurs_possibles, key="pyr_val")
            subset_df = subset_df[subset_df[mode_filtre] == val_retenue]
            titre_complement = f"{mode_filtre} : {val_retenue}"
        
        st.markdown("#### 🧭 Indicateurs de Vigilance")
        kpi1, kpi2 = st.columns(2)
        if 'AGE_CALC' in subset_df.columns:
            total_pop = len(subset_df)
            if total_pop > 0:
                nb_40plus = len(subset_df[subset_df['AGE_CALC'] >= 40])
                ratio_basc = (nb_40plus / total_pop) * 100
                nb_30minus = len(subset_df[subset_df['AGE_CALC'] <= 30])
                nb_50plus = len(subset_df[subset_df['AGE_CALC'] >= 50])
                ratio_renouv = (nb_30minus / nb_50plus * 100) if nb_50plus > 0 else 0
                
                kpi1.metric("Ratio de Basculement (Age ≥ 40 / Effectif total)", f"{ratio_basc:.1f}%")
                if ratio_basc > 50: kpi1.error("⚠️ Processus de vieillissement (> 50%)")
                else: kpi1.success("Structure jeune")
                
                txt_renouv = f"{ratio_renouv:.1f}%" if nb_50plus > 0 else "N/A (Pas de seniors)"
                kpi2.metric("Ratio de Renouvellement (≤30 / ≥50)", txt_renouv)
                if nb_50plus > 0:
                    if ratio_renouv < 100: kpi2.error("⚠️ Non remplacement des départs (< 100%)")
                    else: kpi2.success("Renouvellement assuré")
        st.markdown("---")
        
        c_unit, c_split = st.columns(2)
        unit_display = c_unit.radio("Unité", ["Effectif (Nb)", "Pourcentage (%)"], horizontal=True)
        has_sex = 'SEXE' in subset_df.columns
        mode_split = c_split.radio("Affichage", ["Global (Ensemble)", "H/F Séparés (Papillon)"], horizontal=True, disabled=not has_sex)
        if not has_sex and mode_split == "H/F Séparés (Papillon)": mode_split = "Global (Ensemble)"

        col_target_pyr = 'AGE_CALC' if var_pyr == "Âge" else 'ANC_CALC'
        if col_target_pyr in subset_df.columns:
            bins = list(range(15, 80, 5)) if var_pyr == "Âge" else list(range(0, 45, 5))
            labels = [f"{i}-{i+4}" for i in bins[:-1]]
            subset_df['Tranche'] = pd.cut(subset_df[col_target_pyr], bins=bins, labels=labels, right=False)
            
            group_cols = ['Tranche']
            if mode_split == "H/F Séparés (Papillon)": group_cols.append('SEXE')
            
            df_g = subset_df.groupby(group_cols, observed=False).size().reset_index(name='Count')
            total_g = df_g['Count'].sum()
            df_g['Percent'] = (df_g['Count'] / total_g) * 100
            val_col = 'Percent' if unit_display == "Pourcentage (%)" else 'Count'
            lbl_axis = "% Effectif" if unit_display == "Pourcentage (%)" else "Effectif"

            fig = go.Figure()
            if mode_split == "Global (Ensemble)":
                fig = px.bar(df_g, x='Tranche', y=val_col, text_auto='.1f' if 'Percent' in val_col else True,
                             title=f"Pyramide {titre_complement} ({yr_pyr})", color_discrete_sequence=['#1f77b4'])
                fig.update_layout(xaxis_title="Tranches", yaxis_title=lbl_axis)
                fig.update_traces(textangle=0)
            else:
                sexes = df_g['SEXE'].unique()
                list_h = [s for s in sexes if str(s).upper().startswith('H') or str(s).upper().startswith('M')]
                list_f = [s for s in sexes if str(s).upper().startswith('F')]
                ordered_sexes = list_h + list_f 
                
                for s in ordered_sexes:
                    d = df_g[df_g['SEXE'] == s]
                    val_plot = d[val_col]
                    col_bar = get_gender_color(s)
                    is_man = str(s).upper().startswith('H') or str(s).upper().startswith('M')
                    if is_man: val_plot = val_plot * -1 
                    
                    fig.add_trace(go.Bar(
                        y=d['Tranche'], x=val_plot, name=str(s), orientation='h', marker_color=col_bar,
                        text=d[val_col].round(1) if 'Percent' in val_col else d[val_col], textangle=0
                    ))
                
                fig.update_layout(
                    title=f"Pyramide {titre_complement} par Sexe ({yr_pyr})", barmode='overlay',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(title=f"{lbl_axis} (Hommes à gauche / Femmes à droite)", tickformat="s"), yaxis=dict(title="Tranches")
                )
            
            st.plotly_chart(fig, use_container_width=True)
            csv_pyr = convert_df(df_g)
            st.download_button("📥 Télécharger (CSV)", data=csv_pyr, file_name='pyramide_data.csv', mime='text/csv')

    elif selection_page == "📍 Cartographie Structures":
        if combined_df.empty: st.warning("Veuillez charger des fichiers Effectifs (Stock)."); st.stop()
        st.header("Cartographie Structurelle")
        c1, c2, c3 = st.columns(3)
        mode_visu = c1.radio("Mode", ["Statique (1 année)", "Dynamique (Évolution)"])
        grp_tri = c2.selectbox("Maille", cat_cols)
        critere = c3.selectbox("Critère", ["Âge", "Ancienneté"])
        st.markdown("---")
        st.markdown("#### ⚙️ Réglage des Axes")
        c_min, c_max = st.columns(2)
        user_min = c_min.number_input("Seuil Minimum (%)", value=-5.0, step=1.0)
        user_max = c_max.number_input("Seuil Maximum (%)", value=100.0, step=5.0)
        st.markdown("---")

        cs1, cs2, cs3 = st.columns(3)
        if critere == "Âge":
            val_col = 'AGE_CALC'
            s_low = cs1.slider("Âge Max 'Jeunes'", 20, 40, 30)
            s_high = cs2.slider("Âge Min 'Seniors'", 45, 60, 50)
            lbl_x, lbl_y = f"% Seniors (>{s_high} ans)", f"% Jeunes (<{s_low} ans)"
            lbl_short = "d'Âge"
        else:
            val_col = 'ANC_CALC'
            s_low = cs1.slider("Anc. Max 'Nouveaux'", 1, 10, 5)
            s_high = cs2.slider("Anc. Min 'Anciens'", 10, 30, 15)
            lbl_x, lbl_y = f"% Anciens (>{s_high} ans)", f"% Nouveaux (<{s_low} ans)"
            lbl_short = "d'Ancienneté"
        show_labels = cs3.checkbox("Afficher les étiquettes", value=False)

        def get_stats(df_in, group_col, val_col, low, high, min_eff):
            if val_col not in df_in.columns: return pd.DataFrame()
            res = []
            for name, group in df_in.groupby(group_col, observed=True):
                valid_group = group[group[val_col].notna()]
                total = len(valid_group)
                if total >= min_eff:
                    res.append({
                        'Groupe': name, 'Effectif': total,
                        'Pct_Low': (len(valid_group[valid_group[val_col] < low]) / total) * 100,
                        'Pct_High': (len(valid_group[valid_group[val_col] >= high]) / total) * 100,
                        'Full_Name': name
                    })
            return pd.DataFrame(res)
        
        axis_style = dict(range=[user_min, user_max], showline=True, linewidth=1, linecolor='black', mirror=True, showgrid=True, gridcolor='#eee', zeroline=True, zerolinewidth=1, zerolinecolor='grey')

        if mode_visu == "Statique (1 année)":
            y_photo = st.selectbox("Année", sorted_years, index=len(sorted_years)-1, key="tri_yr")
            titre_graph = f"Structure {lbl_short} par {grp_tri} en {y_photo}"
            all_groups = sorted(data_dict[str(y_photo)][grp_tri].dropna().unique())
            c_sel_static, c_all_static = st.columns([3, 1])
            with c_all_static:
                st.write("")
                sel_all_s = st.checkbox("Tout sélectionner", value=False, key="chk_all_stat")
            with c_sel_static:
                if sel_all_s: sel_groups = st.multiselect("Filtrer:", all_groups, default=all_groups, key="ms_stat")
                else: sel_groups = st.multiselect("Filtrer:", all_groups, default=all_groups[:10] if len(all_groups)>10 else all_groups, key="ms_stat")
            
            if sel_groups:
                df_source = data_dict[str(y_photo)]
                df_source = df_source[df_source[grp_tri].isin(sel_groups)]
                df_viz = get_stats(df_source, grp_tri, val_col, s_low, s_high, min_eff_global)
                if not df_viz.empty:
                    fig = px.scatter(df_viz, x='Pct_High', y='Pct_Low', size='Effectif', color='Groupe', hover_name='Full_Name', text='Groupe' if show_labels else None, title=titre_graph, labels={'Pct_High': lbl_x, 'Pct_Low': lbl_y}, size_max=45, color_discrete_sequence=extended_palette)
                    if show_labels: fig.update_traces(textposition='top center')
                    fig.update_layout(height=700, xaxis=axis_style, yaxis=axis_style, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
                    st.plotly_chart(fig, use_container_width=True)

        else: # DYNAMIQUE
            c_deb, c_fin = st.columns(2)
            y_deb = c_deb.selectbox("Départ", sorted_years, index=0, key="tri_d")
            y_fin = c_fin.selectbox("Arrivée", sorted_years, index=len(sorted_years)-1, key="tri_f")
            titre_graph = f"Évolution des structures {lbl_short} entre {y_deb} et {y_fin}"
            grps_1 = set(data_dict[str(y_deb)][grp_tri].dropna().unique())
            grps_2 = set(data_dict[str(y_fin)][grp_tri].dropna().unique())
            common_grps = sorted(list(grps_1 & grps_2))
            c_sel_traj, c_all_traj = st.columns([3, 1])
            with c_all_traj:
                st.write("")
                sel_all = st.checkbox("Tout sélectionner", value=False, key="chk_all_dyn")
            with c_sel_traj:
                if sel_all: sel_traj = st.multiselect("Groupes à comparer:", common_grps, default=common_grps, key="ms_dyn")
                else: sel_traj = st.multiselect("Groupes à comparer:", common_grps, default=common_grps[:5] if len(common_grps)>5 else common_grps, key="ms_dyn")
            
            if sel_traj:
                df_1 = data_dict[str(y_deb)][data_dict[str(y_deb)][grp_tri].isin(sel_traj)]
                df_2 = data_dict[str(y_fin)][data_dict[str(y_fin)][grp_tri].isin(sel_traj)]
                viz_1 = get_stats(df_1, grp_tri, val_col, s_low, s_high, min_eff_global)
                viz_2 = get_stats(df_2, grp_tri, val_col, s_low, s_high, min_eff_global)
                if not viz_1.empty and not viz_2.empty:
                    merged = pd.merge(viz_1, viz_2, on='Groupe', suffixes=('_start', '_end'))
                    fig = go.Figure()
                    for i, row in merged.iterrows():
                        color = extended_palette[i % len(extended_palette)]
                        fig.add_trace(go.Scatter(x=[row['Pct_High_start']], y=[row['Pct_Low_start']], mode='markers', marker=dict(symbol='circle-open', size=10, color=color), name=f"{row['Full_Name_start']}", hovertext=f"{row['Full_Name_start']} ({y_deb})", showlegend=False))
                        mode_point = 'markers+text' if show_labels else 'markers'
                        fig.add_trace(go.Scatter(x=[row['Pct_High_end']], y=[row['Pct_Low_end']], mode=mode_point, marker=dict(symbol='circle', size=12, color=color), text=row['Groupe'] if show_labels else None, textposition='top center', name=f"{row['Full_Name_end']}", hovertext=f"{row['Full_Name_end']} ({y_fin})"))
                        fig.add_annotation(x=row['Pct_High_end'], y=row['Pct_Low_end'], ax=row['Pct_High_start'], ay=row['Pct_Low_start'], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=color)
                    fig.update_layout(title=titre_graph, xaxis_title=lbl_x, yaxis_title=lbl_y, height=700, xaxis=axis_style, yaxis=axis_style, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)) 
                    st.plotly_chart(fig, use_container_width=True)

    elif selection_page == "🧿 Micro-Analyse":
        if combined_df.empty: st.warning("Veuillez charger des fichiers Effectifs (Stock)."); st.stop()
        st.header("🧿 Micro-Analyse : Croisement Âge / Ancienneté")
        c_yr_n, c_filt_n, c_col_n = st.columns(3)
        yr_nuage = c_yr_n.selectbox("Année", sorted_years, index=len(sorted_years)-1, key="nuage_year")
        opts_filter_nuage = ["Global"] + cat_cols
        lvl_filter = c_filt_n.selectbox("Niveau d'analyse", opts_filter_nuage, key="nuage_lvl")
        df_nuage = data_dict[str(yr_nuage)].copy()
        
        selection_val = None
        if lvl_filter != "Global":
            vals = sorted(df_nuage[lvl_filter].dropna().unique())
            selection_val = st.selectbox(f"Choisir {lvl_filter}", vals)
            df_nuage = df_nuage[df_nuage[lvl_filter] == selection_val]
        
        avail_cols = [c for c in df_nuage.columns if c in cat_cols or c == 'SEXE']
        opts_color = ["Aucun (Couleur unique)"] + avail_cols
        
        default_idx = 0
        if lvl_filter == "Global" and "SERVICE" in avail_cols: default_idx = opts_color.index("SERVICE")
        elif lvl_filter == "SERVICE" and "EMPLOI" in avail_cols: default_idx = opts_color.index("EMPLOI")
        
        color_by = c_col_n.selectbox("Colorier les points par", opts_color, index=default_idx, key="nuage_color")
        
        if 'AGE_CALC' in df_nuage.columns and 'ANC_CALC' in df_nuage.columns:
            plot_color = None
            params_color = {}
            if color_by == "Aucun (Couleur unique)": params_color['color_discrete_sequence'] = ['#1f77b4']
            else:
                plot_color = color_by
                if color_by == "SEXE": params_color['color_discrete_map'] = COLOR_MAP_SEXE
                else: params_color['color_discrete_sequence'] = extended_palette
            
            hover_cols = ['SERVICE', 'EMPLOI', 'SEXE', 'CATEGORIE']
            real_hover_cols = [c for c in hover_cols if c in df_nuage.columns]
            group_keys = ['AGE_CALC', 'ANC_CALC'] + real_hover_cols
            if plot_color and plot_color not in group_keys: group_keys.append(plot_color)
            
            df_agg = df_nuage.groupby(group_keys, observed=False).size().reset_index(name='NB_SALARIES')
            titre_nuage = f"Répartition Individuelle ({len(df_nuage)} agents)"
            if selection_val: titre_nuage += f" - {selection_val}"

            fig = px.scatter(df_agg, x='AGE_CALC', y='ANC_CALC', color=plot_color, size='NB_SALARIES', hover_data=real_hover_cols + ['NB_SALARIES'], title=titre_nuage, opacity=0.8, size_max=12, **params_color)
            fig.add_shape(type="line", x0=20, y0=0, x1=65, y1=45, line=dict(color="lightgray", dash="dot"))
            fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>Âge: %{x}<br>Anc: %{y}<br><b>Effectif: %{marker.size}</b>')
            fig.update_layout(xaxis_title="Âge", yaxis_title="Ancienneté", height=650)
            st.plotly_chart(fig, use_container_width=True)

    elif selection_page == "🚪 Analyse des Sorties (Turn-over)":
        st.header("🚪 Analyse des Sorties (Turn-over)")
        
        if df_sorties.empty:
            st.info("ℹ️ Veuillez charger le fichier 'Sorties' dans le menu latéral (Zone 2) pour activer cet onglet.")
        else:
            if 'ANNEE_SORTIE' in df_sorties.columns:
                years_available = sorted(df_sorties['ANNEE_SORTIE'].dropna().unique())
                tabs_sorties = st.tabs(["📉 Évolution Temporelle", "🥧 Répartition / Structure", "🧿 Micro-Analyse (Nuage de points)"])
                
                with tabs_sorties[0]:
                    st.subheader("Évolution du nombre de départs par an")
                    c1, c2 = st.columns(2)
                    dims_evo = ["Global", "SERVICE", "EMPLOI", "SEXE", "MOTIF", "CONTRAT"]
                    dims_evo = [d for d in dims_evo if d == "Global" or d in df_sorties.columns]
                    dim_evo = c1.selectbox("Axe d'analyse", dims_evo)
                    if dim_evo == "Global":
                        df_evo = df_sorties.groupby("ANNEE_SORTIE").size().reset_index(name="Nombre de Sorties")
                        fig = px.bar(df_evo, x="ANNEE_SORTIE", y="Nombre de Sorties", text="Nombre de Sorties", title="Évolution globale des sorties", color_discrete_sequence=['#d62728']) 
                    else:
                        df_evo = df_sorties.groupby(["ANNEE_SORTIE", dim_evo]).size().reset_index(name="Nombre de Sorties")
                        params_col = {}
                        if dim_evo == "SEXE": params_col['color_discrete_map'] = COLOR_MAP_SEXE
                        else: params_col['color_discrete_sequence'] = extended_palette
                        fig = px.bar(df_evo, x="ANNEE_SORTIE", y="Nombre de Sorties", color=dim_evo, title=f"Évolution des sorties par {dim_evo}", **params_col)
                    fig.update_layout(xaxis=dict(tickmode='linear'), xaxis_title="Année de Sortie")
                    st.plotly_chart(fig, use_container_width=True)

                with tabs_sorties[1]:
                    st.subheader("Répartition et Structure des Sorties")
                    c1, c2 = st.columns(2)
                    yrs_sel = c1.multiselect("Années à inclure", years_available, default=years_available)
                    dims_rep = ["SERVICE", "EMPLOI", "SEXE", "MOTIF", "CONTRAT", "TRANCHE_AGE", "TRANCHE_ANC"]
                    dims_rep = [d for d in dims_rep if d in df_sorties.columns]
                    dim_rep = c2.selectbox("Critère de répartition", dims_rep)
                    if yrs_sel:
                        df_sub = df_sorties[df_sorties['ANNEE_SORTIE'].isin(yrs_sel)]
                        if not df_sub.empty:
                            df_count = df_sub.groupby(dim_rep).size().reset_index(name="Count")
                            type_chart = st.radio("Type de graphique", ["Camembert (Pie)", "Barres"], horizontal=True)
                            params_col = {}
                            if dim_rep == "SEXE": params_col['color_discrete_map'] = COLOR_MAP_SEXE
                            else: params_col['color_discrete_sequence'] = extended_palette
                            if type_chart == "Camembert (Pie)": fig = px.pie(df_count, names=dim_rep, values="Count", title=f"Répartition par {dim_rep}", **params_col)
                            else: fig = px.bar(df_count, x=dim_rep, y="Count", color=dim_rep, title=f"Répartition par {dim_rep}", **params_col)
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("##### Données détaillées"); st.dataframe(df_count, hide_index=True)
                        else: st.warning("Aucune donnée pour les années sélectionnées.")
                    else: st.info("Sélectionnez au moins une année.")
                
                with tabs_sorties[2]:
                    st.subheader("Micro-Analyse : Caractéristiques des Sortants (Âge vs Ancienneté)")
                    if 'AGE_SORTIE' in df_sorties.columns and 'ANC_SORTIE' in df_sorties.columns:
                        c_yrs_m, c_grp_m, c_col_m = st.columns(3)
                        yrs_micro = c_yrs_m.multiselect("Années à analyser", years_available, default=years_available, key="micro_s_yr")
                        dims_filter = ["Global", "SERVICE", "EMPLOI", "MOTIF", "SEXE"]
                        dims_filter = [d for d in dims_filter if d == "Global" or d in df_sorties.columns]
                        lvl_filter = c_grp_m.selectbox("Filtrer la population", dims_filter, key="micro_s_lvl")
                        df_m = df_sorties[df_sorties['ANNEE_SORTIE'].isin(yrs_micro)].copy()
                        sel_val_m = None
                        if lvl_filter != "Global":
                            vals_m = sorted(df_m[lvl_filter].dropna().unique())
                            sel_val_m = st.selectbox(f"Choisir {lvl_filter}", vals_m, key="micro_s_val")
                            df_m = df_m[df_m[lvl_filter] == sel_val_m]
                        avail_cols_m = [c for c in df_m.columns if c in cat_cols or c == 'MOTIF']
                        opts_color_m = ["Aucun (Couleur unique)"] + avail_cols_m
                        color_by_m = c_col_m.selectbox("Colorier les points par", opts_color_m, key="micro_s_col")
                        if not df_m.empty:
                            df_m['Age_R'] = df_m['AGE_SORTIE'].round()
                            df_m['Anc_R'] = df_m['ANC_SORTIE'].round()
                            params_color_m = {}
                            plot_color_m = None
                            if color_by_m == "Aucun (Couleur unique)": params_color_m['color_discrete_sequence'] = ['#d62728']
                            else:
                                plot_color_m = color_by_m
                                if color_by_m == "SEXE": params_color_m['color_discrete_map'] = COLOR_MAP_SEXE
                                else: params_color_m['color_discrete_sequence'] = extended_palette
                            grp_keys_m = ['Age_R', 'Anc_R']
                            hover_cols_m = ['SERVICE', 'EMPLOI', 'MOTIF', 'SEXE']
                            if plot_color_m and plot_color_m not in grp_keys_m: grp_keys_m.append(plot_color_m)
                            df_agg_m = df_m.groupby(grp_keys_m, observed=False).size().reset_index(name='Count')
                            real_hover_m = [c for c in hover_cols_m if c in df_agg_m.columns]
                            fig_m = px.scatter(df_agg_m, x='Age_R', y='Anc_R', size='Count', color=plot_color_m, title=f"Profil des Sorties ({len(df_m)} départs)", hover_data=real_hover_m + ['Count'], opacity=0.7, size_max=15, **params_color_m)
                            fig_m.add_shape(type="line", x0=20, y0=0, x1=65, y1=45, line=dict(color="lightgray", dash="dot"))
                            fig_m.update_traces(hovertemplate='<b>%{hovertext}</b><br>Âge: %{x}<br>Anc: %{y}<br><b>Sorties: %{marker.size}</b>')
                            fig_m.update_layout(xaxis_title="Âge au départ", yaxis_title="Ancienneté au départ", height=600)
                            st.plotly_chart(fig_m, use_container_width=True)
                        else: st.warning("Aucune donnée avec ces filtres.")
                    else: st.error("Données d'âge ou d'ancienneté manquantes (Colonnes NAISSANCE/ENTREE).")
            else: st.error("Le fichier 'Sorties' ne contient pas de date de sortie valide.")

    elif selection_page == "📈 Évolution Effectifs":
        if combined_df.empty: st.warning("Veuillez charger des fichiers Effectifs (Stock)."); st.stop()
        st.header("📈 Évolution des Effectifs")
        c_filt_eff, c_sub_eff = st.columns(2)
        opts_eff = ["Global"] + cat_cols
        mode_eff = c_filt_eff.selectbox("Vue", opts_eff, key="eff_view")
        df_eff_viz = combined_df.copy()
        titre_eff = "Évolution de l'Effectif Global"
        if mode_eff != "Global":
            vals_eff = sorted(df_eff_viz[mode_eff].dropna().unique())
            sel_eff = c_sub_eff.selectbox(f"Choisir {mode_eff}", vals_eff, key="eff_sel")
            df_eff_viz = df_eff_viz[df_eff_viz[mode_eff] == sel_eff]
            titre_eff = f"Évolution Effectif : {sel_eff}"
        counts = df_eff_viz.groupby("ANNEE_FICH").size().reset_index(name="Effectif")
        fig = px.bar(counts, x="ANNEE_FICH", y="Effectif", text="Effectif", title=titre_eff)
        fig.update_xaxes(type='category')
        st.plotly_chart(fig, use_container_width=True)

    elif selection_page == "📋 Types de Contrat":
        if combined_df.empty: st.warning("Veuillez charger des fichiers Effectifs (Stock)."); st.stop()
        st.header("📋 Analyse des Types de Contrat")
        if 'CONTRAT' in combined_df.columns:
            c_filt_ctr, c_sub_ctr = st.columns(2)
            opts_ctr = ["Global"] + cat_cols
            mode_ctr = c_filt_ctr.selectbox("Vue", opts_ctr, key="ctr_view")
            df_ctr_viz = combined_df.copy()
            titre_ctr = "Répartition des Contrats (Global)"
            if mode_ctr != "Global":
                vals_ctr = sorted(df_ctr_viz[mode_ctr].dropna().unique())
                sel_ctr = c_sub_ctr.selectbox(f"Choisir {mode_ctr}", vals_ctr, key="ctr_sel")
                df_ctr_viz = df_ctr_viz[df_ctr_viz[mode_ctr] == sel_ctr]
                titre_ctr = f"Répartition des Contrats : {sel_ctr}"
            df_grp = df_ctr_viz.groupby(['ANNEE_FICH', 'CONTRAT']).size().reset_index(name='Count')
            fig = px.bar(df_grp, x="ANNEE_FICH", y="Count", color="CONTRAT", title=titre_ctr, color_discrete_sequence=extended_palette)
            fig.update_layout(barmode='stack', barnorm='percent', yaxis_title="% Effectif")
            fig.update_xaxes(type='category')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Colonne 'TYPE CONTRAT' ou 'STATUT' non trouvée dans les fichiers.")

    elif selection_page == "📉 Flux (Histo Décalé)":
        if combined_df.empty: st.warning("Veuillez charger des fichiers Effectifs (Stock)."); st.stop()
        st.header("Histogrammes décalés")
        c_var, c_start, c_end = st.columns(3)
        var_analyse = c_var.radio("Axe", ["Âge", "Ancienneté"], horizontal=True, key="flux_axe")
        y_start = c_start.selectbox("Année Base (Passé)", sorted_years, index=0, key="flux_start")
        y_end = c_end.selectbox("Année Cible (Réel)", sorted_years, index=len(sorted_years)-1, key="flux_end")
        st.markdown("---")
        c_filt_type, c_filt_val = st.columns(2)
        filter_opts = ["Effectif Global"]
        if 'SERVICE' in combined_df.columns: filter_opts.append("Service")
        if 'EMPLOI' in combined_df.columns: filter_opts.append("Emploi")
        filter_mode = c_filt_type.selectbox("Filtrer la population par :", filter_opts, key="flux_filtre")
        selected_val = None
        if filter_mode != "Effectif Global":
            map_col = {"Service": "SERVICE", "Emploi": "EMPLOI"}
            col_target = map_col[filter_mode]
            df_p_temp = data_dict[str(y_start)]
            vals_p = set(df_p_temp[col_target].dropna().unique()) if col_target in df_p_temp.columns else set()
            selected_val = c_filt_val.selectbox(f"Choisir {filter_mode} :", sorted(list(vals_p)), key="flux_val")
        if y_start and y_end:
            shift = int(y_end) - int(y_start)
            df_past = data_dict[str(y_start)].copy()
            df_curr = data_dict[str(y_end)].copy()
            if selected_val:
                col_target = map_col[filter_mode]
                if col_target in df_past.columns: df_past = df_past[df_past[col_target] == selected_val]
                if col_target in df_curr.columns: df_curr = df_curr[df_curr[col_target] == selected_val]
            ready = False
            if var_analyse == "Âge" and 'AGE_CALC' in df_past.columns:
                df_past['VAL_PROJ'] = df_past['AGE_CALC'] + shift; df_curr['VAL_REEL'] = df_curr['AGE_CALC']
                label_x = "Âge (ans)"; ready = True
            elif var_analyse == "Ancienneté" and 'ANC_CALC' in df_past.columns:
                df_past['VAL_PROJ'] = df_past['ANC_CALC'] + shift; df_curr['VAL_REEL'] = df_curr['ANC_CALC']
                label_x = "Ancienneté (ans)"; ready = True
            if ready:
                idx_range = range(0, 75)
                vc_proj = df_past['VAL_PROJ'].value_counts().reindex(idx_range, fill_value=0)
                vc_reel = df_curr['VAL_REEL'].value_counts().reindex(idx_range, fill_value=0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=vc_proj.index, y=vc_proj.values, name=f"Théorique ({y_start} + {shift} ans)", line=dict(color='orange', width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=vc_reel.index, y=vc_reel.values, name=f"Réel ({y_end})", fill='tozeroy', line=dict(color='#1f77b4', width=3)))
                fig.update_layout(title=f"Histogramme Décalé - {filter_mode}", xaxis_title=label_x, yaxis_title="Effectif")
                st.plotly_chart(fig, use_container_width=True)

    elif selection_page == "📊 Absentéisme (Évolution)":
        if combined_df.empty: st.warning("Veuillez charger des fichiers Effectifs (Stock)."); st.stop()
        st.header("Diagnostic Absentéisme : Évolution des Poids")
        keywords_abs = ['NB J', 'NB H', 'ABS', 'NB AT', 'NB MP', 'NB D\'AT', 'ACCIDENT', 'MALADIE']
        all_abs_cols = [c for c in combined_df.columns if any(k in c for k in keywords_abs)]
        abs_cols_clean = [c for c in all_abs_cols if "FORMATION" not in c]
        if abs_cols_clean:
            st.info("💡 Comparez le poids d'un service dans l'effectif global vs son poids dans l'absentéisme.")
            c_yrs_abs, c_grp_abs, c_ind_abs = st.columns(3)
            years_abs = c_yrs_abs.multiselect("Années à analyser", sorted_years, default=sorted_years, key="abs_yrs_img")
            grp_abs = c_grp_abs.selectbox("Axe (Service, Métier...)", cat_cols, key="abs_grp_img")
            ind_abs = c_ind_abs.selectbox("Indicateur (Motif)", abs_cols_clean, key="abs_ind_img")
            if years_abs and grp_abs and ind_abs:
                df_period = combined_df[combined_df['ANNEE_FICH'].isin([int(y) for y in years_abs])]
                eff_total = len(df_period)
                df_eff_dist = df_period.groupby(grp_abs).size().reset_index(name='Count')
                df_eff_dist['Percentage'] = (df_eff_dist['Count'] / eff_total) * 100
                df_eff_dist['Category'] = "1. % Part Effectif"
                final_data = [df_eff_dist[[grp_abs, 'Percentage', 'Category']]]
                for y in sorted(years_abs):
                    df_y = combined_df[combined_df['ANNEE_FICH'] == int(y)]
                    total_abs_y = df_y[ind_abs].sum()
                    if total_abs_y > 0:
                        df_abs_dist = df_y.groupby(grp_abs)[ind_abs].sum().reset_index(name='Sum_Abs')
                        df_abs_dist['Percentage'] = (df_abs_dist['Sum_Abs'] / total_abs_y) * 100
                        df_abs_dist['Category'] = f"% Part {ind_abs} {y}"
                        final_data.append(df_abs_dist[[grp_abs, 'Percentage', 'Category']])
                df_viz_abs = pd.concat(final_data, ignore_index=True)
                fig = px.bar(df_viz_abs, x="Category", y="Percentage", color=grp_abs, title=f"Évolution des poids : Effectif vs {ind_abs}", color_discrete_sequence=extended_palette)
                fig.update_layout(barmode='stack', yaxis_title="Part (%)", xaxis_title="", legend_traceorder="reversed")
                st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Aucune donnée d'absentéisme pertinente détectée.")

    elif selection_page == "⚖️ Absentéisme (Comparaison)":
        if combined_df.empty: st.warning("Veuillez charger des fichiers Effectifs (Stock)."); st.stop()
        st.header("⚖️ Répartition de l'Absentéisme par Motif")
        keywords_abs = ['NB J', 'NB H', 'ABS', 'NB AT', 'NB MP', 'NB D\'AT', 'ACCIDENT', 'MALADIE']
        all_abs_cols = [c for c in combined_df.columns if any(k in c for k in keywords_abs)]
        abs_cols_clean = [c for c in all_abs_cols if "FORMATION" not in c]
        if abs_cols_clean:
            c1, c2 = st.columns(2)
            years_comp = c1.multiselect("Années (Cumul)", sorted_years, default=sorted_years, key="abs_comp_yr")
            grp_comp = c2.selectbox("Axe d'analyse", cat_cols, key="abs_comp_grp")
            if years_comp:
                df_c = combined_df[combined_df['ANNEE_FICH'].isin([int(y) for y in years_comp])].copy()
                plot_data = []
                tmp_eff = df_c.groupby(grp_comp, observed=True).size().reset_index(name='Valeur')
                total_eff = tmp_eff['Valeur'].sum()
                tmp_eff['Pct'] = (tmp_eff['Valeur'] / total_eff) * 100; tmp_eff['Indicateur'] = "1. % Effectif"
                plot_data.append(tmp_eff[[grp_comp, 'Pct', 'Indicateur']])
                for m in abs_cols_clean:
                    tmp_abs = df_c.groupby(grp_comp, observed=True)[m].sum().reset_index(name='Valeur')
                    total_abs = tmp_abs['Valeur'].sum()
                    if total_abs > 0:
                        tmp_abs['Pct'] = (tmp_abs['Valeur'] / total_abs) * 100; tmp_abs['Indicateur'] = m
                        plot_data.append(tmp_abs[[grp_comp, 'Pct', 'Indicateur']])
                if plot_data:
                    df_plot = pd.concat(plot_data, ignore_index=True)
                    fig = px.bar(df_plot, x="Indicateur", y="Pct", color=grp_comp, title=f"Poids des services sur l'absentéisme (Cumul {min(years_comp)}-{max(years_comp)})", color_discrete_sequence=extended_palette)
                    fig.update_layout(barmode='stack', barnorm='percent', yaxis_title="Part (%)", legend_traceorder="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning("Pas de données sur la période.")
        else: st.warning("Pas de colonnes d'absentéisme.")

    elif selection_page == "💰 Autres (Promo/Form/Rém)":
        if combined_df.empty: st.warning("Veuillez charger des fichiers Effectifs (Stock)."); st.stop()
        st.header("Analyse des autres indicateurs (Sur-représentation)")
        other_keywords = ['PROMO', 'FORMATION', 'REMUN', 'SALAIRE', 'AUGMENTATION']
        other_cols = [c for c in combined_df.columns if any(k in c for k in other_keywords)]
        if other_cols:
            c1, c2 = st.columns(2)
            years_oth = c1.multiselect("Années (Cumul)", sorted_years, default=sorted_years, key="oth_yr")
            grp_oth = c2.selectbox("Axe d'analyse", cat_cols, key="oth_grp")
            if years_oth:
                df_c = combined_df[combined_df['ANNEE_FICH'].isin([int(y) for y in years_oth])].copy()
                plot_data = []
                tmp_eff = df_c.groupby(grp_oth, observed=True).size().reset_index(name='Valeur')
                total_eff = tmp_eff['Valeur'].sum()
                tmp_eff['Pct'] = (tmp_eff['Valeur'] / total_eff) * 100; tmp_eff['Indicateur'] = "1. % Effectif"
                plot_data.append(tmp_eff[[grp_oth, 'Pct', 'Indicateur']])
                for m in other_cols:
                    tmp_kpi = df_c.groupby(grp_oth, observed=True)[m].sum().reset_index(name='Valeur')
                    total_kpi = tmp_kpi['Valeur'].sum()
                    if total_kpi > 0:
                        tmp_kpi['Pct'] = (tmp_kpi['Valeur'] / total_kpi) * 100; tmp_kpi['Indicateur'] = m
                        plot_data.append(tmp_kpi[[grp_oth, 'Pct', 'Indicateur']])
                if plot_data:
                    df_plot = pd.concat(plot_data, ignore_index=True)
                    fig = px.bar(df_plot, x="Indicateur", y="Pct", color=grp_oth, title=f"Poids relatifs (Cumul {min(years_oth)}-{max(years_oth)})", color_discrete_sequence=extended_palette)
                    fig.update_layout(barmode='stack', barnorm='percent', yaxis_title="Part (%)", legend_traceorder="reversed")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.warning("Pas de données non nulles sur la période.")
        else: st.info("Aucune colonne type 'Formation', 'Promotion' ou 'Rémunération' trouvée.")

    elif selection_page == "📝 Restrictions Santé":
        if combined_df.empty: st.warning("Veuillez charger des fichiers Effectifs (Stock)."); st.stop()
        st.header("📝 Analyse Qualitative : Restrictions Santé & Verbatims")
        if 'RESTRICTION' in combined_df.columns:
            opts_yr = sorted_years + ["Toutes les années (Cumul)"]
            c1, c2, c3 = st.columns(3)
            yr_qual = c1.selectbox("Année", opts_yr, index=len(opts_yr)-2 if len(opts_yr)>1 else 0, key="qual_yr")
            if yr_qual == "Toutes les années (Cumul)": df_qual = combined_df.copy()
            else: df_qual = data_dict[str(yr_qual)].copy()
            df_qual = df_qual[df_qual['RESTRICTION'].notna() & (df_qual['RESTRICTION'].astype(str).str.strip() != "")]
            grp_qual = c2.selectbox("Filtrer par (Optionnel)", ["Global"] + cat_cols, key="qual_grp")
            if grp_qual != "Global":
                val_qual = c3.selectbox(f"Choisir {grp_qual}", sorted(df_qual[grp_qual].dropna().unique()), key="qual_val")
                df_qual = df_qual[df_qual[grp_qual] == val_qual]
            st.markdown(f"**Nombre de verbatims trouvés : {len(df_qual)}**")
            cols_show = ['ANNEE_FICH', 'SERVICE', 'EMPLOI', 'RESTRICTION']
            if 'SEXE' in df_qual.columns: cols_show.insert(3, 'SEXE')
            st.dataframe(df_qual[cols_show], use_container_width=True, hide_index=True)
            csv = convert_df(df_qual[cols_show])
            st.download_button("📥 Télécharger les verbatims (CSV)", data=csv, file_name='verbatims_restrictions.csv', mime='text/csv')
        else: st.info("Aucune colonne de type 'RESTRICTION', 'AVIS' ou 'APTITUDE' trouvée dans le fichier.")

else:
    st.info("👈 Veuillez charger vos fichiers Excel ou CSV dans le menu à gauche.")
    st.markdown("---")
    st.markdown("### Besoin d'aide ou de modèles ?")
    st.markdown("Allez dans l'onglet **📚 Ressources & Guides** (même sans données chargées) pour accéder aux fiches pratiques.")
