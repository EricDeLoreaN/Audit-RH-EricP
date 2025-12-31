import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Audit Social RH", layout="wide")

# ==========================================
# 🔒 SÉCURITÉ : GESTION DU MOT DE PASSE
# ==========================================
MOT_DE_PASSE = "ARACTNORMANDIERH"  # <--- VOTRE MOT DE PASSE

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    if st.session_state.password_input == MOT_DE_PASSE:
        st.session_state.authenticated = True
        del st.session_state.password_input # On nettoie la variable pour sécurité
    else:
        st.error("Mot de passe incorrect ❌")

if not st.session_state.authenticated:
    st.title("🔒 Accès Restreint")
    st.markdown("### Outil d'Analyse Démographique & Absentéisme")
    st.text_input("Veuillez saisir le mot de passe :", type="password", key="password_input", on_change=check_password)
    st.stop()  # ⛔ LE SCRIPT S'ARRÊTE ICI SI PAS CONNECTÉ

# ==========================================
# 📂 FONCTIONS DE CHARGEMENT ROBUSTES
# ==========================================
@st.cache_data
def load_data(file):
    """
    Tente de lire le fichier qu'il soit Excel ou CSV,
    même si l'extension est fausse.
    """
    # 1. Essai lecture Excel standard
    try:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip().str.upper()
        return df
    except Exception:
        pass

    # 2. Essai lecture CSV (Séparateur Point-Virgule - Format Excel FR)
    try:
        file.seek(0)
        df = pd.read_csv(file, sep=';', encoding='latin-1', on_bad_lines='skip')
        if len(df.columns) > 1:
            df.columns = df.columns.str.strip().str.upper()
            return df
    except Exception:
        pass

    # 3. Essai lecture CSV (Séparateur Virgule - Format Standard)
    try:
        file.seek(0)
        df = pd.read_csv(file, sep=',', encoding='utf-8', on_bad_lines='skip')
        if len(df.columns) > 1:
            df.columns = df.columns.str.strip().str.upper()
            return df
    except Exception:
        pass

    return None

def extract_year(filename):
    """Trouve une année de 4 chiffres dans le nom du fichier"""
    match = re.search(r'20\d{2}', filename)
    return int(match.group(0)) if match else None

# ==========================================
# 🚀 DÉBUT DE L'APPLICATION
# ==========================================
st.title("📊 Analyse Sociale & Absentéisme")
st.markdown("Données chargées et sécurisées.")

# --- SIDEBAR : IMPORTATION ---
st.sidebar.header("1. Données Sources")
uploaded_files = st.sidebar.file_uploader("Chargez vos fichiers (Bases annuelles)", accept_multiple_files=True)

data_dict = {}
combined_df = pd.DataFrame()

if uploaded_files:
    temp_dfs = []
    for f in uploaded_files:
        year = extract_year(f.name)
        if year:
            df = load_data(f)
            if df is not None:
                # --- NETTOYAGE ET RENOMMAGE ---
                # On harmonise les colonnes CDC / ARACT
                rename_map = {
                    'SERVICE / SECTEUR': 'SERVICE', 
                    'SECTEUR': 'SERVICE', 
                    'UNITÉ': 'SERVICE',
                    'AGENCE': 'SERVICE',
                    'DURÉE DU TRAVAIL': 'TEMPS_TRAVAIL',
                    'ARRIVÉE': 'ENTREE', 
                    'DATE ENTREE': 'ENTREE',
                    'DATE D\'ENTRÉE': 'ENTREE',
                    'NAISSANCE': 'NAISSANCE',
                    'DATE NAISSANCE': 'NAISSANCE'
                }
                # Renommage uniquement si la colonne existe
                cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
                df = df.rename(columns=cols_to_rename)
                
                df['ANNEE_FICH'] = year
                data_dict[str(year)] = df
                temp_dfs.append(df)
            else:
                st.sidebar.error(f"❌ Illisible : {f.name}")
        else:
            st.sidebar.warning(f"⚠️ Pas d'année dans le nom : {f.name}")
    
    if temp_dfs:
        combined_df = pd.concat(temp_dfs, ignore_index=True)
        sorted_years = sorted(data_dict.keys())
        st.sidebar.success(f"✅ {len(data_dict)} années : {min(sorted_years)} à {max(sorted_years)}")
        
        # --- PRÉ-CALCULS GLOBAUX ---
        # Identification des colonnes clés après renommage
        cols = combined_df.columns
        col_naiss = next((c for c in cols if 'NAISS' in c), None)
        col_entree = next((c for c in cols if 'ENTREE' in c), None)
        
        # Identification Colonnes Catégorielles
        possible_cols = ['SERVICE', 'EMPLOI', 'SEXE', 'CATEGORIE', 'POSTE']
        cat_cols = [c for c in possible_cols if c in combined_df.columns]

        if col_naiss and col_entree:
             # Conversion Dates (dayfirst=True pour gérer le format FR 31/12/2020)
             combined_df['Date_Naiss'] = pd.to_datetime(combined_df[col_naiss], dayfirst=True, errors='coerce')
             combined_df['Date_Entree'] = pd.to_datetime(combined_df[col_entree], dayfirst=True, errors='coerce')
             
             # Calcul Age et Ancienneté au moment du fichier
             combined_df['AGE_CALC'] = combined_df['ANNEE_FICH'] - combined_df['Date_Naiss'].dt.year
             combined_df['ANC_CALC'] = combined_df['ANNEE_FICH'] - combined_df['Date_Entree'].dt.year
             
             # Nettoyage des valeurs absurdes
             combined_df = combined_df[(combined_df['AGE_CALC'] > 14) & (combined_df['AGE_CALC'] < 80)]
        else:
            st.error(f"⚠️ Colonnes NAISSANCE ou ENTREE introuvables. Colonnes vues : {list(cols)}")

# ==========================================
# 📊 VISUALISATIONS
# ==========================================
if not combined_df.empty:
    tabs = st.tabs(["📉 Flux (Histo Décalé)", "📍 Structure (Nuage)", "📊 Absentéisme (Barres 100%)"])

    # --- TAB 1 : FLUX (CREAPT) ---
    with tabs[0]:
        st.header("Analyse des Flux : Projection vs Réalité")
        
        c_var, c_start, c_end = st.columns(3)
        var_analyse = c_var.radio("Axe d'analyse", ["Âge", "Ancienneté"], horizontal=True)
        y_start = c_start.selectbox("Année Base (Passé)", sorted_years, index=0)
        y_end = c_end.selectbox("Année Cible (Réel)", sorted_years, index=len(sorted_years)-1)
        
        if y_start and y_end:
            shift = int(y_end) - int(y_start)
            st.info(f"Décalage de {shift} ans entre {y_start} et {y_end}")
            
            df_past = data_dict[y_start].copy()
            df_curr = data_dict[y_end].copy()
            
            # Calcul des projections
            # On utilise les mêmes colonnes détectées plus haut
            col_n = next((c for c in df_past.columns if 'NAISS' in c), None)
            col_e = next((c for c in df_past.columns if 'ENTREE' in c), None)

            if var_analyse == "Âge":
                # Age théorique = (Année départ - Naissance) + Décalage
                df_past['VAL_PROJ'] = (int(y_start) - pd.to_datetime(df_past[col_n], dayfirst=True, errors='coerce').dt.year) + shift
                df_curr['VAL_REEL'] = int(y_end) - pd.to_datetime(df_curr[col_n], dayfirst=True, errors='coerce').dt.year
                label_x = "Âge (ans)"
            else:
                # Ancienneté théorique
                df_past['VAL_PROJ'] = (int(y_start) - pd.to_datetime(df_past[col_e], dayfirst=True, errors='coerce').dt.year) + shift
                df_curr['VAL_REEL'] = int(y_end) - pd.to_datetime(df_curr[col_e], dayfirst=True, errors='coerce').dt.year
                label_x = "Ancienneté (ans)"
            
            # Agrégation et Alignement
            idx_range = range(0, 75)
            vc_proj = df_past['VAL_PROJ'].value_counts().reindex(idx_range, fill_value=0)
            vc_reel = df_curr['VAL_REEL'].value_counts().reindex(idx_range, fill_value=0)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vc_proj.index, y=vc_proj.values, name=f"Théorique (Effectif {y_start} + {shift} ans)", line=dict(color='orange', width=2, dash='dash')))
            fig.add_trace(go.Scatter(x=vc_reel.index, y=vc_reel.values, name=f"Réel (Effectif {y_end})", fill='tozeroy', line=dict(color='#1f77b4', width=3)))
            fig.update_layout(title="Histogramme Décalé", xaxis_title=label_x, yaxis_title="Effectif", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2 : STRUCTURE ---
    with tabs[1]:
        st.header("Cartographie des Collectifs")
        
        c1, c2, c3 = st.columns(3)
        y_tri = c1.selectbox("Année", sorted_years, index=len(sorted_years)-1, key="tri_y")
        grp_tri = c2.selectbox("Groupe", cat_cols, key="tri_g")
        mode_tri = c3.selectbox("Critères", ["Générationnel (<30 vs >50)", "Ancienneté (<5 vs >15)"])
        
        df_t = data_dict[y_tri].copy()
        
        # Calculs variables temporaires pour le tri
        col_n_t = next((c for c in df_t.columns if 'NAISS' in c), None)
        col_e_t = next((c for c in df_t.columns if 'ENTREE' in c), None)
        
        if "Générationnel" in mode_tri:
            df_t['VAL'] = int(y_tri) - pd.to_datetime(df_t[col_n_t], dayfirst=True, errors='coerce').dt.year
            th_low, th_high = 30, 50
            lbl_y, lbl_x = "% Jeunes (<30 ans)", "% Seniors (>50 ans)"
        else:
            df_t['VAL'] = int(y_tri) - pd.to_datetime(df_t[col_e_t], dayfirst=True, errors='coerce').dt.year
            th_low, th_high = 5, 15
            lbl_y, lbl_x = "% Nouveaux (<5 ans)", "% Anciens (>15 ans)"

        stats = []
        for n, g in df_t.groupby(grp_tri):
            if len(g) >= 3: # Filtre petits groupes
                stats.append({
                    'Groupe': n, 
                    'Effectif': len(g),
                    'Pct_Low': (len(g[g['VAL'] < th_low]) / len(g)) * 100,
                    'Pct_High': (len(g[g['VAL'] >= th_high]) / len(g)) * 100
                })
        
        if stats:
            df_viz = pd.DataFrame(stats)
            fig = px.scatter(df_viz, x='Pct_High', y='Pct_Low', size='Effectif', color='Groupe', text='Groupe',
                             title=f"Positionnement par {grp_tri}", labels={'Pct_High': lbl_x, 'Pct_Low': lbl_y}, size_max=60)
            # Lignes médianes pour aider la lecture
            fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="lightgray", dash="dot"))
            fig.add_shape(type="line", x0=0, y0=50, x1=100, y1=50, line=dict(color="lightgray", dash="dot"))
            
            fig.update_traces(textposition='top center')
            fig.update_layout(xaxis=dict(range=[-5, 105]), yaxis=dict(range=[-5, 105]))
            st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3 : ABSENTÉISME (Barres 100%) ---
    with tabs[2]:
        st.header("Comparatif de Poids (Effectif vs Absentéisme)")
        
        col_param, col_graph = st.columns([1, 3])
        with col_param:
            years_abs = st.multiselect("Années (Cumul)", sorted_years, default=sorted_years)
            grp_abs = st.selectbox("Axe d'analyse", cat_cols + ['TR_AGE', 'TR_ANC'], index=0, key="abs_g")
            
            # Détection automatique des colonnes d'absence (NB J...)
            abs_cols = [c for c in combined_df.columns if 'NB J' in c or 'NB H' in c or 'ABS' in c]
            metrics_abs = st.multiselect("Indicateurs", abs_cols, default=abs_cols[:2] if abs_cols else None)

        if years_abs and grp_abs and metrics_abs:
            df_c = combined_df[combined_df['ANNEE_FICH'].isin([int(y) for y in years_abs])].copy()
            
            # Création des tranches si demandé
            if grp_abs == 'TR_AGE':
                df_c['TR_AGE'] = pd.cut(df_c['AGE_CALC'], bins=[0,25,35,45,55,100], labels=['<25','25-35','35-45','45-55','55+'])
            elif grp_abs == 'TR_ANC':
                df_c['TR_ANC'] = pd.cut(df_c['ANC_CALC'], bins=[0,2,5,10,20,100], labels=['<2','2-5','5-10','10-20','20+'])

            plot_data = []
            
            # 1. Poids Effectif
            tmp_eff = df_c.groupby(grp_abs).size().reset_index(name='Valeur')
            tmp_eff['Indicateur'] = "1. Poids Effectif"
            plot_data.append(tmp_eff)
            
            # 2. Poids Absences
            for m in metrics_abs:
                tmp_abs = df_c.groupby(grp_abs)[m].sum().reset_index()
                tmp_abs.columns = [grp_abs, 'Valeur'] # Renommage propre
                tmp_abs['Indicateur'] = f"2. {m}"
                plot_data.append(tmp_abs)
            
            df_plot = pd.concat(plot_data, ignore_index=True)
            
            with col_graph:
                fig = px.bar(df_plot, x="Indicateur", y="Valeur", color=grp_abs, 
                             title=f"Répartition Normalisée (100%) par {grp_abs}", 
                             text_auto='.1f')
                
                fig.update_layout(
                    barmode='stack', 
                    barnorm='percent', # <--- C'est ça qui met tout à 100%
                    yaxis_title="Part (%)",
                    xaxis_title="",
                    height=600
                )
                fig.update_traces(hovertemplate='%{y:.1f}%<br>%{x}')
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Si un segment de couleur s'élargit vers la droite, le groupe est sur-représenté dans l'absentéisme.")

else:
    st.info("👈 Veuillez charger vos fichiers Excel ou CSV dans le menu latéral.")
