import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Football Scout", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .stMetric {
        background-color: rgba(29, 185, 84, 0.05); 
        padding: 10px; 
        border-radius: 10px; 
        border-left: 4px solid #1DB954;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    h1 {
        font-weight: 800;
        background: -webkit-linear-gradient(#1DB954, #121212);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stContainer {
        padding: 15px;
        border-radius: 10px;
        background-color: #fcfcfc;
        margin-bottom: 15px;
        border: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DİL SEÇİMİ VE SÖZLÜK / LANGUAGE SELECTION & DICTIONARY ---
lang = st.sidebar.radio("🌐 Language / Dil", ["🇬🇧 English", "🇹🇷 Türkçe"])
is_tr = (lang == "🇹🇷 Türkçe")

t = {
    "title": "🧠 AI Odaklı Küresel Scout Platformu" if is_tr else "🧠 AI-Driven Global Scout Platform",
    "subtitle": "Kulüp felsefesine uygun, makine öğrenmesi destekli oyuncu keşif ve analiz sistemi." if is_tr else "Machine learning-powered player discovery and analysis system tailored to club philosophy.",
    "tab1": "Ana Scout Ekranı (Teknik)" if is_tr else "Main Scout Screen (Technical)",
    "tab2": "Fiziksel Analiz Modülü (Demo)" if is_tr else "Physical Analysis Module (Demo)",
    "filters": "🔍 Scout Filtreleri" if is_tr else "🔍 Scout Filters",
    "philosophy": "Kulüp Oyun Felsefesi:" if is_tr else "Club Playing Philosophy:",
    "position": "Aranacak Mevkiyi Seçin:" if is_tr else "Select Position:",
    "count": "Gösterilecek Oyuncu Sayısı:" if is_tr else "Number of Recommendations:",
    "target": "Hedef Oyuncuyu Seçin:" if is_tr else "Select Target Player:",
    "target_profile": "🎯 Hedef Profil:" if is_tr else "🎯 Target Profile:",
    "team_league": "Takım & Lig" if is_tr else "Team & League",
    "age_pos": "Yaş & Pozisyon" if is_tr else "Age & Position",
    "tech_kpi": "Teknik KPI (Uyum)" if is_tr else "Technical KPI (Fit)",
    "phys_kpi": "Fiziksel KPI" if is_tr else "Physical KPI",
    "awaiting_data": "Veri bekleniyor" if is_tr else "Awaiting data",
    "radar_title": "### 📊 Karşılaştırmalı Radar Analizi (Yüzdelik Dilim)" if is_tr else "### 📊 Comparative Radar Analysis (Percentile)",
    "radar_desc": "Grafikteki değerler, oyuncunun kendi mevki havuzundaki yüzdelik dilimini gösterir. Yüzeye ne kadar yayılırsa o kadar iyidir." if is_tr else "Values in the chart show the player's percentile rank within their position pool. The larger the area, the better.",
    "alternatives": "Felsefesine Uygun Alternatifler" if is_tr else "Alternatives Fitting the Philosophy",
    "trend": "trend" if is_tr else "trend",
    "perf_trend": "Sezonluk Performans Trendi (Maç Başı)" if is_tr else "Seasonal Performance Trend (Per 90)",
    "err_data": "Veri seti bulunamadı. Lütfen CSV dosyalarını kontrol edin." if is_tr else "Dataset not found. Please check CSV files.",
    "err_ml": "Bu filtrelerde makine öğrenmesi analizi yapacak yeterli oyuncu bulunamadı." if is_tr else "Not enough players found in these filters to run ML analysis.",
    "phys_title": "🏃 Fiziksel Analiz Modülü (Doğrulama Ortamı)" if is_tr else "🏃 Physical Analysis Module (Validation Environment)",
    "phys_info": "Not: Bu modül, gerçek oyuncu verilerine sahte fiziksel veriler atamamak için 'Anonim Oyuncular' üzerinde çalışır." if is_tr else "Note: This module runs on 'Anonymous Players' to avoid assigning fake physical stats to real players. It will be integrated into the Main Screen once licensed data is available.",
    "anon_data": "### Anonim Veri Seti" if is_tr else "### Anonymous Dataset",
    "dist": "Koşu Mesafesi" if is_tr else "Distance Covered",
    "speed": "Sprint / Max Hız" if is_tr else "Sprints / Max Speed"
}

stat_isimleri = {
    'Gls': 'Gol' if is_tr else 'Goals',
    'Ast': 'Asist' if is_tr else 'Assists',
    'Sh': 'Şut' if is_tr else 'Shots',
    'SoT': 'İsab. Şut' if is_tr else 'Shots on Target',
    'TklW': 'Top Kapma' if is_tr else 'Tackles Won',
    'Int': 'Pas Arası' if is_tr else 'Interceptions',
    'Crs': 'Orta Açma' if is_tr else 'Crosses',
    'Fld': 'Faul Alma' if is_tr else 'Fouls Drawn',
    'Fls': 'Faul Yapma' if is_tr else 'Fouls Committed',
    'Off': 'Ofsayt' if is_tr else 'Offsides'
}

felsefeler_tr = ["Dengeli (Varsayılan)", "Yüksek Pres", "Topa Sahip Olma", "Kontra Atak"]
felsefeler_en = ["Balanced (Default)", "High Press", "Possession", "Counter Attack"]
felsefeler_list = felsefeler_tr if is_tr else felsefeler_en

felsefeler_map = {
    felsefeler_list[0]: {'Gls': 0.15, 'Ast': 0.15, 'Sh': 0.1, 'TklW': 0.2, 'Int': 0.2, 'Crs': 0.1, 'Fld': 0.1},
    felsefeler_list[1]: {'TklW': 0.4, 'Int': 0.4, 'Fls': 0.2},
    felsefeler_list[2]: {'Ast': 0.5, 'Crs': 0.3, 'Fld': 0.2},
    felsefeler_list[3]: {'Gls': 0.4, 'Sh': 0.3, 'SoT': 0.2, 'Off': 0.1}
}

mevki_sozlugu = {
    ("Tüm Mevkiler" if is_tr else "All Positions"): "",
    ("Forvetler (Hücum)" if is_tr else "Forwards (Attack)"): "FW",
    ("Orta Sahalar (Merkez)" if is_tr else "Midfielders (Center)"): "MF",
    ("Defanslar (Savunma)" if is_tr else "Defenders (Defense)"): "DF"
}

# --- ARAYÜZ / UI ---
st.title(t["title"])
st.markdown(t["subtitle"])
st.markdown("---")

tab1, tab2 = st.tabs([t["tab1"], t["tab2"]])

with tab1:
    @st.cache_data
    def tum_veriyi_hazirla():
        def sezonu_isle(dosya_adi, etiket):
            df = pd.read_csv(dosya_adi)
            df = df[df['Min'] >= 500].copy()
            df['90s'] = pd.to_numeric(df['90s'], errors='coerce').fillna(1)
            df = df[df['90s'] > 0]
            
            tum_istatistikler = ['Gls', 'Ast', 'Sh', 'SoT', 'TklW', 'Int', 'Crs', 'Fld', 'Fls', 'Off']
            for stat in tum_istatistikler:
                if stat in df.columns:
                    df[stat] = pd.to_numeric(df[stat], errors='coerce').fillna(0)
                    df[f'{stat}_per90_{etiket}'] = df[stat] / df['90s']
                else:
                    df[f'{stat}_per90_{etiket}'] = 0.0
                
            lazim = ['Player', 'Squad', 'Comp', 'Pos', 'Age', '90s'] + [f'{stat}_per90_{etiket}' for stat in tum_istatistikler]
            return df[lazim]

        try:
            df1 = sezonu_isle("players_data-2024_2025.csv", "Eski")
            df2 = sezonu_isle("players_data-2025_2026.csv", "Yeni")
        except Exception:
            return pd.DataFrame(), []
        
        df = pd.merge(df1, df2, on='Player', how='inner', suffixes=('_eski', '_yeni'))
        df['Squad'] = df['Squad_yeni']
        df['Comp'] = df['Comp_yeni']
        df['Pos'] = df['Pos_yeni']
        df['Age'] = df['Age_yeni']
        
        tum_istatistikler = ['Gls', 'Ast', 'Sh', 'SoT', 'TklW', 'Int', 'Crs', 'Fld', 'Fls', 'Off']
        for stat in tum_istatistikler:
            df[f'{stat}_farki'] = df[f'{stat}_per90_Yeni'] - df[f'{stat}_per90_Eski']
            
        lig_agirliklari = {'Premier League': 1.0, 'La Liga': 0.95, 'Serie A': 0.95, 'Bundesliga': 0.90, 'Ligue 1': 0.85}
        def katsayi_bul(lig):
            if not isinstance(lig, str): return 0.7
            for l, k in lig_agirliklari.items():
                if l in lig: return k
            return 0.7
            
        df['Lig_Katsayisi'] = df['Comp'].apply(katsayi_bul)
        return df.reset_index(drop=True), tum_istatistikler

    df_genel, tum_istatistikler = tum_veriyi_hazirla()

    if len(df_genel) == 0:
        st.error(t["err_data"])
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.header(t["filters"])
    secilen_felsefe = st.sidebar.selectbox(t["philosophy"], felsefeler_list)

    if secilen_felsefe in ["Yüksek Pres", "High Press"]:
        msg = "💡 **Scout Analist Notu:** Halka açık verilerdeki kısıtlamalar nedeniyle Yüksek Pres hesaplamasında temsilci (proxy) olarak 'Top Kapma' ve 'Pas Arası' kullanılmıştır. Sistem, Premium (Opta/StatsBomb) veriler entegre edildiğinde doğrudan **'Başarılı Baskı (Successful Pressures)'** ve **'PPDA'** metriklerini hesaplayacak modülerliktedir." if is_tr else "💡 **Scout Analyst Note:** Due to public data limitations, 'Tackles Won' and 'Interceptions' are used as proxy metrics for High Press. When integrated with Premium event data, this engine is designed to instantly utilize **'Successful Pressures'** and **'PPDA'** metrics."
        st.info(msg)

    secilen_mevki_etiket = st.sidebar.selectbox(t["position"], list(mevki_sozlugu.keys()))
    mevki_kodu = mevki_sozlugu[secilen_mevki_etiket]
    oneri_sayisi = st.sidebar.slider(t["count"], min_value=1, max_value=20, value=5)

    if mevki_kodu:
        df_filtrelenmis = df_genel[df_genel['Pos'].str.contains(mevki_kodu, na=False)].reset_index(drop=True)
    else:
        df_filtrelenmis = df_genel.copy()

    aktif_istatistikler = tum_istatistikler
    knn_ozellikleri = []
    
    for stat in aktif_istatistikler:
        df_filtrelenmis[f'{stat}_yeni_ag'] = df_filtrelenmis[f'{stat}_per90_Yeni'] * df_filtrelenmis['Lig_Katsayisi']
        knn_ozellikleri.append(f'{stat}_yeni_ag')
        df_filtrelenmis[f'{stat}_pct'] = df_filtrelenmis[f'{stat}_per90_Yeni'].rank(pct=True) * 100

    kpi_scaler = MinMaxScaler(feature_range=(0, 100))
    kpi_agirliklari = felsefeler_map[secilen_felsefe]

    if len(df_filtrelenmis) > 5:
        scaled_kpi_features = kpi_scaler.fit_transform(df_filtrelenmis[[f'{stat}_per90_Yeni' for stat in kpi_agirliklari.keys()]])
        scaled_kpi_df = pd.DataFrame(scaled_kpi_features, columns=kpi_agirliklari.keys())
        
        df_filtrelenmis['Uyum_Puani'] = 0
        for stat, weight in kpi_agirliklari.items():
            df_filtrelenmis['Uyum_Puani'] += scaled_kpi_df[stat] * weight
            
        df_filtrelenmis['Uyum_Puani'] = df_filtrelenmis['Uyum_Puani'] * df_filtrelenmis['Lig_Katsayisi']
        df_filtrelenmis['Uyum_Puani'] = (df_filtrelenmis['Uyum_Puani'] / df_filtrelenmis['Uyum_Puani'].max()) * 100

        scaler = StandardScaler()
        olcekli_veri = scaler.fit_transform(df_filtrelenmis[knn_ozellikleri])
        model = NearestNeighbors(n_neighbors=len(df_filtrelenmis), algorithm='auto')
        model.fit(olcekli_veri)

        oyuncu_listesi = df_filtrelenmis['Player'].sort_values().tolist()
        secilen_oyuncu = st.sidebar.selectbox(t["target"], oyuncu_listesi)

        if secilen_oyuncu:
            hedef_idx = df_filtrelenmis[df_filtrelenmis['Player'] == secilen_oyuncu].index[0]
            hedef = df_filtrelenmis.iloc[hedef_idx]
            
            st.markdown(f"## {t['target_profile']} {hedef['Player']}")
            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            col_h1.metric(t["team_league"], f"{hedef['Squad']}", f"{hedef['Comp']}")
            col_h2.metric(t["age_pos"], f"{int(hedef['Age'])}", f"{hedef['Pos']}")
            col_h3.metric(t["tech_kpi"], f"{hedef['Uyum_Puani']:.1f}/100")
            col_h4.metric(t["phys_kpi"], "—", t["awaiting_data"])
            
            mesafeler, benzerler = model.kneighbors([olcekli_veri[hedef_idx]])
            aday_indexleri = benzerler[0][1:oneri_sayisi*5]
            adaylar = df_filtrelenmis.iloc[aday_indexleri].copy()
            adaylar = adaylar.sort_values(by='Uyum_Puani', ascending=False).head(oneri_sayisi)
            
            kpi_gostergeler = list(kpi_agirliklari.keys())
            radar_kategoriler = [stat_isimleri.get(s, s) for s in kpi_gostergeler]
            
            en_iyi_aday = adaylar.iloc[0] if len(adaylar) > 0 else None
            
            st.markdown("---")
            st.markdown(t["radar_title"])
            st.caption(t["radar_desc"])
            
            if en_iyi_aday is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[hedef[f'{s}_pct'] for s in kpi_gostergeler],
                    theta=radar_kategoriler,
                    fill='toself',
                    name=hedef['Player'],
                    line_color='#1DB954'
                ))
                fig.add_trace(go.Scatterpolar(
                    r=[en_iyi_aday[f'{s}_pct'] for s in kpi_gostergeler],
                    theta=radar_kategoriler,
                    fill='toself',
                    name=en_iyi_aday['Player'],
                    line_color='#ff4b4b'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=True,
                    margin=dict(l=40, r=40, t=40, b=40),
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown(f"### 🤖 {secilen_felsefe} {t['alternatives']}")
            
            for idx, onerilen in adaylar.iterrows():
                with st.expander(f"⭐ {onerilen['Player']} | {onerilen['Squad']} | {onerilen['Comp']} ({t['tech_kpi']}: {onerilen['Uyum_Puani']:.1f}/100)"):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1:
                        st.markdown(f"**Age:** {int(onerilen['Age'])}")
                        st.markdown(f"**Position:** {onerilen['Pos']}")
                    with c2:
                        st.metric(t["tech_kpi"], f"{onerilen['Uyum_Puani']:.1f}/100")
                        st.metric(t["phys_kpi"], "—")
                    with c3:
                        st.markdown(f"**{t['perf_trend']}**")
                        trend_cols = st.columns(len(kpi_gostergeler))
                        for j, stat in enumerate(kpi_gostergeler):
                            trend_cols[j].metric(f"{stat_isimleri.get(stat, stat)}", f"{onerilen[f'{stat}_per90_Yeni']:.2f}", f"{onerilen[f'{stat}_farki']:.2f}")

    else:
        st.warning(t["err_ml"])


with tab2:
    st.subheader(t["phys_title"])
    st.info(t["phys_info"])
    
    if os.path.exists("physical_mock_data.csv"):
        df_phys = pd.read_csv("physical_mock_data.csv")
        
        st.write(t["anon_data"])
        st.dataframe(df_phys, use_container_width=True)
        
        scaler_phys = MinMaxScaler(feature_range=(0, 100))
        phys_features = ['Total_Distance_km', 'High_Speed_Run_m', 'Sprint_Count', 'Max_Speed_kmh']
        scaled_phys = scaler_phys.fit_transform(df_phys[phys_features])
        df_scaled_phys = pd.DataFrame(scaled_phys, columns=phys_features)
        
        idx = felsefeler_list.index(secilen_felsefe)
        if idx == 1: # High Press
            phys_weights = {'Total_Distance_km': 0.2, 'High_Speed_Run_m': 0.4, 'Sprint_Count': 0.4, 'Max_Speed_kmh': 0.0}
        elif idx == 3: # Counter Attack
            phys_weights = {'Total_Distance_km': 0.1, 'High_Speed_Run_m': 0.3, 'Sprint_Count': 0.3, 'Max_Speed_kmh': 0.3}
        else: # Possession / Balanced
            phys_weights = {'Total_Distance_km': 0.4, 'High_Speed_Run_m': 0.2, 'Sprint_Count': 0.2, 'Max_Speed_kmh': 0.2}
            
        df_phys['Fiziksel_KPI'] = 0
        for stat, w in phys_weights.items():
            df_phys['Fiziksel_KPI'] += df_scaled_phys[stat] * w
            
        df_phys['Fiziksel_KPI'] = (df_phys['Fiziksel_KPI'] / df_phys['Fiziksel_KPI'].max()) * 100
        
        df_phys_sorted = df_phys.sort_values(by="Fiziksel_KPI", ascending=False)
        for _, row in df_phys_sorted.iterrows():
            with st.container():
                cols = st.columns([2, 1, 1, 1])
                cols[0].markdown(f"**{row['Player_ID']}** ({row['Pos']})")
                cols[1].metric(t["phys_kpi"], f"{row['Fiziksel_KPI']:.1f}/100")
                cols[2].metric(t["dist"], f"{row['Total_Distance_km']} km")
                cols[3].metric(t["speed"], f"{row['Sprint_Count']} / {row['Max_Speed_kmh']} km/h")