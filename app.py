import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Diagnostic Flash RH_Eric PELTIER", layout="wide")

# ==========================================
# 🔒 SÉCURITÉ
# ==========================================
MOT_DE_PASSE = "ericpeltier"

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def check_password():
    if st.session_state.password_input == MOT_DE_PASSE:
        st.session_state.authenticated = True
        del st.session_state.password_input
    else:
        st.error("Mot de passe incorrect ❌")

if not st.session_state.authenticated:
    st.title("🔒 Accès Restreint")
    st.markdown("### Outil Diagnostic Flash RH_Eric PELTIER")
    st.text_input("Veuillez saisir le mot de passe :", type="password", key="password_input", on_change=check_password)
    st.stop()

# ==========================================
# 📂 CHARGEMENT ROBUSTE
# ==========================================
@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file)
        df.columns = df.columns.str.strip().str.upper()
        return df
    except Exception:
        pass
    try:
        file.seek(0)
        df = pd.read_csv(file, sep=';', encoding='latin-1', on_bad_lines='skip')
        if len(df.columns) > 1:
            df.columns = df.columns.str.strip().str.upper()
            return df
    except Exception:
        pass
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
    match = re.search(r'20\d{2}', filename)
    return int(match.group(0)) if match else None

# --- PALETTE DE COULEURS ÉTENDUE ---
# Fusion de plusieurs palettes pour avoir plus de nuances
extended_palette = (
    px.colors.qualitative.G10 + 
    px.colors.qualitative.T10 + 
    px.colors.qualitative.Bold + 
    px.colors.qualitative.Vivid + 
    px.colors.qualitative.Alphabet + 
    px.colors.qualitative.Dark24
)

# ==========================================
# 🚀 APPLICATION
# ==========================================
st.title("📊 Diagnostic Flash RH_Eric PELTIER")

# --- SIDEBAR ---
st.sidebar.header("1. Données Sources")
uploaded_files = st.sidebar.file_uploader("Chargez vos fichiers", accept_multiple_files=True)

min_eff_global = st.sidebar.slider("Taille minimale d'un groupe pour affichage", 1, 20, 3, help="Les services ayant moins de X personnes seront masqués.")

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
                    'NAISSANCE': 'NAISSANCE', 'DATE NAISSANCE': 'NAISSANCE'
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
        st.sidebar.success(f"✅ {len(data_dict)} années chargées")
        cat_cols = [c for c in ['SERVICE', 'EMPLOI', 'SEXE', 'CATEGORIE', 'POSTE'] if c in combined_df.columns]
    else:
        sorted_years = []

# ==========================================
# 📊 VISUALISATIONS
# ==========================================
if not combined_df.empty:
    tabs = st.tabs(["📉 Flux (Histo Décalé)", "📍 Structure & Trajectoire", "📊 Absentéisme"])

    # --- TAB 1 : FLUX ---
    with tabs[0]:
        st.header("Histogrammes décalés (Théorique Vs Réel)")
        
        c_var, c_start, c_end = st.columns(3)
        var_analyse = c_var.radio("Axe", ["Âge", "Ancienneté"], horizontal=True)
        y_start = c_start.selectbox("Année Base (Passé)", sorted_years, index=0)
        y_end = c_end.selectbox("Année Cible (Réel)", sorted_years, index=len(sorted_years)-1)
        
        st.markdown("---")
        c_filt_type, c_filt_val = st.columns(2)
        
        filter_opts = ["Effectif Global"]
        if 'SERVICE' in combined_df.columns: filter_opts.append("Service")
        if 'EMPLOI' in combined_df.columns: filter_opts.append("Emploi")
        if 'SEXE' in combined_df.columns: filter_opts.append("Sexe")
        
        filter_mode = c_filt_type.selectbox("Filtrer la population par :", filter_opts)
        
        selected_val = None
        if filter_mode != "Effectif Global":
            map_col = {"Service": "SERVICE", "Emploi": "EMPLOI", "Sexe": "SEXE"}
            col_target = map_col[filter_mode]
            
            df_p_temp = data_dict[str(y_start)]
            df_c_temp = data_dict[str(y_end)]
            vals_p = set(df_p_temp[col_target].dropna().unique()) if col_target in df_p_temp.columns else set()
            vals_c = set(df_c_temp[col_target].dropna().unique()) if col_target in df_c_temp.columns else set()
            possible_vals = sorted(list(vals_p | vals_c))
            
            selected_val = c_filt_val.selectbox(f"Choisir {filter_mode} :", possible_vals)

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
                df_past['VAL_PROJ'] = df_past['AGE_CALC'] + shift
                df_curr['VAL_REEL'] = df_curr['AGE_CALC']
                label_x = "Âge (ans)"
                ready = True
            elif var_analyse == "Ancienneté" and 'ANC_CALC' in df_past.columns:
                df_past['VAL_PROJ'] = df_past['ANC_CALC'] + shift
                df_curr['VAL_REEL'] = df_curr['ANC_CALC']
                label_x = "Ancienneté (ans)"
                ready = True
            
            if ready:
                if df_past.empty and df_curr.empty:
                    st.warning(f"Aucune donnée trouvée pour {selected_val}.")
                else:
                    idx_range = range(0, 75)
                    vc_proj = df_past['VAL_PROJ'].value_counts().reindex(idx_range, fill_value=0)
                    vc_reel = df_curr['VAL_REEL'].value_counts().reindex(idx_range, fill_value=0)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=vc_proj.index, y=vc_proj.values, name=f"Théorique ({y_start} + {shift} ans)", line=dict(color='orange', width=2, dash='dash')))
                    fig.add_trace(go.Scatter(x=vc_reel.index, y=vc_reel.values, name=f"Réel ({y_end})", fill='tozeroy', line=dict(color='#1f77b4', width=3)))
                    
                    title_graph = f"Histogramme Décalé - {filter_mode}"
                    if selected_val: title_graph += f" : {selected_val}"
                        
                    fig.update_layout(title=title_graph, xaxis_title=label_x, yaxis_title="Effectif", hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données manquantes (Age/Ancienneté).")

    # --- TAB 2 : STRUCTURE & TRAJECTOIRES ---
    with tabs[1]:
        st.header("Diagrammes triangulaires (Comparer les structures d'âge ou d'ancienneté et leurs évolutions dans le temps)")
        
        c1, c2, c3 = st.columns(3)
        mode_visu = c1.radio("Visualisation", ["Photo (1 année)", "Trajectoire (Comparaison)"])
        grp_tri = c2.selectbox("Maille d'analyse", cat_cols)
        critere = c3.selectbox("Critère", ["Générationnel (Âge)", "Ancienneté"])

        st.markdown("---")
        cs1, cs2 = st.columns(2)
        if critere == "Générationnel (Âge)":
            val_col = 'AGE_CALC'
            s_low = cs1.slider("Âge Max 'Jeunes'", 20, 40, 30)
            s_high = cs2.slider("Âge Min 'Seniors'", 45, 60, 50)
            lbl_x, lbl_y = f"% Seniors (>{s_high} ans)", f"% Jeunes (<{s_low} ans)"
        else:
            val_col = 'ANC_CALC'
            s_low = cs1.slider("Anc. Max 'Nouveaux'", 1, 10, 5)
            s_high = cs2.slider("Anc. Min 'Anciens'", 10, 30, 15)
            lbl_x, lbl_y = f"% Anciens (>{s_high} ans)", f"% Nouveaux (<{s_low} ans)"

        def get_stats(df_in, group_col, val_col, low, high, min_eff):
            if val_col not in df_in.columns: return pd.DataFrame()
            res = []
            for name, group in df_in.groupby(group_col, observed=True):
                valid_group = group[group[val_col].notna()]
                total = len(valid_group)
                if total >= min_eff:
                    res.append({
                        'Groupe': name,
                        'Effectif': total,
                        'Pct_Low': (len(valid_group[valid_group[val_col] < low]) / total) * 100,
                        'Pct_High': (len(valid_group[valid_group[val_col] >= high]) / total) * 100
                    })
            return pd.DataFrame(res)

        st.markdown("---")
        
        if mode_visu == "Photo (1 année)":
            y_photo = st.selectbox("Année", sorted_years, index=len(sorted_years)-1)
            all_groups = sorted(data_dict[str(y_photo)][grp_tri].dropna().unique())
            
            col_sel, col_chk = st.columns([3, 1])
            with col_chk:
                st.write("") 
                st.write("") 
                select_all_photo = st.checkbox("Tout sélectionner", value=False, key="all_photo")
            
            with col_sel:
                if select_all_photo:
                    sel_groups = st.multiselect("Filtrer les groupes :", all_groups, default=all_groups)
                else:
                    default_sel = all_groups[:10] if len(all_groups) > 10 else all_groups
                    sel_groups = st.multiselect("Filtrer les groupes :", all_groups, default=default_sel)
            
            if sel_groups:
                df_source = data_dict[str(y_photo)]
                df_source = df_source[df_source[grp_tri].isin(sel_groups)]
                df_viz = get_stats(df_source, grp_tri, val_col, s_low, s_high, min_eff_global)
                
                if not df_viz.empty:
                    fig = px.scatter(df_viz, x='Pct_High', y='Pct_Low', size='Effectif', color='Groupe', text='Groupe',
                                     title=f"Carte {y_photo}", labels={'Pct_High': lbl_x, 'Pct_Low': lbl_y}, size_max=60,
                                     color_discrete_sequence=extended_palette) # PALETTE ETENDUE
                    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="lightgray", dash="dot"))
                    fig.add_shape(type="line", x0=0, y0=50, x1=100, y1=50, line=dict(color="lightgray", dash="dot"))
                    fig.update_traces(textposition='top center')
                    fig.update_layout(xaxis=dict(range=[-5, 105]), yaxis=dict(range=[-5, 105]))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Aucun groupe ne dépasse {min_eff_global} personnes.")

        else: # MODE TRAJECTOIRE
            c_deb, c_fin = st.columns(2)
            y_deb = c_deb.selectbox("Départ", sorted_years, index=0)
            y_fin = c_fin.selectbox("Arrivée", sorted_years, index=len(sorted_years)-1)
            
            grps_1 = set(data_dict[str(y_deb)][grp_tri].dropna().unique())
            grps_2 = set(data_dict[str(y_fin)][grp_tri].dropna().unique())
            common_grps = sorted(list(grps_1 & grps_2))
            
            col_sel_traj, col_chk_traj = st.columns([3, 1])
            with col_chk_traj:
                st.write("")
                st.write("")
                select_all_traj = st.checkbox("Tout sélectionner", value=False, key="all_traj")
            
            with col_sel_traj:
                if select_all_traj:
                    sel_traj = st.multiselect("Choisir les groupes à comparer :", common_grps, default=common_grps)
                else:
                    default_traj = common_grps[:5] if len(common_grps) > 5 else common_grps
                    sel_traj = st.multiselect("Choisir les groupes à comparer :", common_grps, default=default_traj)
            
            if sel_traj:
                df_1 = data_dict[str(y_deb)][data_dict[str(y_deb)][grp_tri].isin(sel_traj)]
                df_2 = data_dict[str(y_fin)][data_dict[str(y_fin)][grp_tri].isin(sel_traj)]
                
                viz_1 = get_stats(df_1, grp_tri, val_col, s_low, s_high, min_eff_global)
                viz_2 = get_stats(df_2, grp_tri, val_col, s_low, s_high, min_eff_global)
                
                if not viz_1.empty and not viz_2.empty:
                    merged = pd.merge(viz_1, viz_2, on='Groupe', suffixes=('_start', '_end'))
                    
                    fig = go.Figure()
                    for i, row in merged.iterrows():
                        color = extended_palette[i % len(extended_palette)] # PALETTE ETENDUE
                        fig.add_trace(go.Scatter(x=[row['Pct_High_start']], y=[row['Pct_Low_start']], mode='markers',
                                                 marker=dict(symbol='circle-open', size=10, color=color), name=f"{row['Groupe']} ({y_deb})", showlegend=False))
                        fig.add_trace(go.Scatter(x=[row['Pct_High_end']], y=[row['Pct_Low_end']], mode='markers+text',
                                                 marker=dict(symbol='circle', size=12, color=color), text=row['Groupe'], textposition='top center',
                                                 name=f"{row['Groupe']} ({y_fin})", showlegend=False))
                        fig.add_annotation(x=row['Pct_High_end'], y=row['Pct_Low_end'], ax=row['Pct_High_start'], ay=row['Pct_Low_start'],
                                           xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=color)
                    
                    fig.update_layout(title=f"Evolution {y_deb} -> {y_fin}", xaxis_title=lbl_x, yaxis_title=lbl_y,
                                      xaxis=dict(range=[-5, 105]), yaxis=dict(range=[-5, 105]), height=700)
                    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color="lightgray", dash="dot"))
                    fig.add_shape(type="line", x0=0, y0=50, x1=100, y1=50, line=dict(color="lightgray", dash="dot"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Données insuffisantes pour les groupes sélectionnés.")

    # --- TAB 3 : ABSENTÉISME ---
    with tabs[2]:
        st.header("Certaines catégories de populations sont-elles sur-représentées dans l'absentéisme au regard de leurs poids dans l'effectif ?")
        col_param, col_graph = st.columns([1, 3])
        with col_param:
            years_abs = st.multiselect("Années (Cumul)", sorted_years, default=sorted_years)
            opts_grp = list(cat_cols)
            if 'AGE_CALC' in combined_df.columns: opts_grp.append('TR_AGE')
            if 'ANC_CALC' in combined_df.columns: opts_grp.append('TR_ANC')
            grp_abs = st.selectbox("Axe d'analyse", opts_grp, index=0, key="abs_g")
            abs_cols = [c for c in combined_df.columns if 'NB J' in c or 'NB H' in c or 'ABS' in c]
            metrics_abs = st.multiselect("Indicateurs", abs_cols, default=abs_cols[:2] if abs_cols else None)

            # FILTRE AVEC OPTION "TOUS"
            sel_abs = None
            if grp_abs in cat_cols:
                uniques = sorted(combined_df[grp_abs].dropna().unique())
                
                st.write("")
                select_all_abs = st.checkbox("Tout sélectionner", value=False, key="all_abs")
                
                if select_all_abs:
                    sel_abs = st.multiselect("Filtrer les groupes :", uniques, default=uniques)
                else:
                    default_abs = uniques[:10] if len(uniques) > 10 else uniques
                    sel_abs = st.multiselect("Filtrer les groupes :", uniques, default=default_abs)

        if years_abs and grp_abs and metrics_abs:
            df_c = combined_df[combined_df['ANNEE_FICH'].isin([int(y) for y in years_abs])].copy()
            
            if sel_abs is not None:
                df_c = df_c[df_c[grp_abs].isin(sel_abs)]

            calc_ok = True
            if grp_abs == 'TR_AGE':
                if 'AGE_CALC' in df_c.columns:
                    df_c['TR_AGE'] = pd.cut(df_c['AGE_CALC'], bins=[0,25,35,45,55,100], labels=['<25','25-35','35-45','45-55','55+'])
                else: calc_ok = False
            elif grp_abs == 'TR_ANC':
                if 'ANC_CALC' in df_c.columns:
                    df_c['TR_ANC'] = pd.cut(df_c['ANC_CALC'], bins=[0,2,5,10,20,100], labels=['<2','2-5','5-10','10-20','20+'])
                else: calc_ok = False

            if calc_ok:
                plot_data = []
                tmp_eff = df_c.groupby(grp_abs, observed=True).size().reset_index(name='Valeur')
                tmp_eff['Indicateur'] = "1. Poids Effectif"
                plot_data.append(tmp_eff)
                for m in metrics_abs:
                    tmp_abs = df_c.groupby(grp_abs, observed=True)[m].sum().reset_index()
                    tmp_abs.columns = [grp_abs, 'Valeur']
                    tmp_abs['Indicateur'] = f"2. {m}"
                    plot_data.append(tmp_abs)
                
                df_plot = pd.concat(plot_data, ignore_index=True)
                with col_graph:
                    fig = px.bar(df_plot, x="Indicateur", y="Valeur", color=grp_abs, 
                                 title=f"Répartition Normalisée (100%) par {grp_abs}", 
                                 text_auto='.1f',
                                 color_discrete_sequence=extended_palette) # PALETTE ETENDUE
                    
                    # C'EST ICI QUE LA MAGIE OPERE POUR LA LEGENDE
                    fig.update_layout(
                        barmode='stack', 
                        barnorm='percent', 
                        yaxis_title="Part (%)", 
                        xaxis_title="", 
                        height=600,
                        legend_traceorder="reversed" # ALIGNE LA LEGENDE SUR L'ORDRE VISUEL
                    )
                    
                    fig.update_traces(hovertemplate='%{y:.1f}%<br>%{x}')
                    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Veuillez charger vos fichiers Excel ou CSV.")
