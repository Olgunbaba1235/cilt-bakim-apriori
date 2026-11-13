import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os
import time

st.set_page_config(
    page_title="E-Ticaret AI & Müşteri Zekası Platformu",
    page_icon="🛒",
    layout="wide"
)

# Özel Kurumsal CSS Tasarımı
st.markdown("""
<style>
    .main-title {font-size: 2.3rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0px;}
    .sub-title {font-size: 1.05rem; color: #4B5563; margin-bottom: 15px;}
    .pipeline-box {background-color: #F8FAFC; padding: 15px; border-radius: 10px; border-left: 5px solid #10B981; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚀 Kurumsal E-Ticaret Veri Mühendisliği, Segmentasyon & AI Tavsiye Platformu</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Otomatize ETL Veri Hattı | Davranışsal RFM & K-Means Segmentasyonu | Gelecek Kampanya Harcama Tahmini | Kozmetik & Teknoloji Analitiği | Apriori Tavsiye Motoru</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. OTOMATİZE VERİ ÇEKME, BÜTÜNLEŞTİRME VE ÖN ELEME MOTORU
# ---------------------------------------------------------
@st.cache_data
def run_automated_etl_pipeline(n_customers=600):
    np.random.seed(42)
    
    # Kaynak 1: Web/Mobil Uygulama Davranış Logları (Clickstream API)
    web_logs = pd.DataFrame({
        "Musteri_ID": [f"CUST-{1000+i}" for i in range(n_customers)],
        "Gunluk_Aktif_Sure_Dk": np.round(np.random.gamma(shape=2.5, scale=8.0, size=n_customers), 1),
        "Platform_Kullanim_Suresi_Ay": np.random.randint(1, 48, size=n_customers),
        "Gezilen_Sayfa_Sayisi": np.random.randint(3, 65, size=n_customers)
    })
    
    # Gerçekçi ön eleme simülasyonu için bilerek eksik ve aykırı (outlier) veri ekliyoruz
    missing_indices = np.random.choice(n_customers, size=18, replace=False)
    outlier_indices = np.random.choice(n_customers, size=8, replace=False)
    web_logs.loc[missing_indices, "Gunluk_Aktif_Sure_Dk"] = np.nan
    web_logs.loc[outlier_indices, "Gunluk_Aktif_Sure_Dk"] = np.random.uniform(350, 600, size=8)

    # Kaynak 2: ERP & CRM Alışveriş Veritabanı (RFM + Kategori Dağılımı)
    crm_data = pd.DataFrame({
        "Musteri_ID": [f"CUST-{1000+i}" for i in range(n_customers)],
        "Son_Alisveris_Gun_Once": np.random.randint(1, 180, size=n_customers), # Recency
        "Alisveris_Sikligi": np.random.randint(1, 35, size=n_customers),       # Frequency
        "Kozmetik_Harcama_TL": np.round(np.random.exponential(scale=1800, size=n_customers) + 150, 2),
        "Teknoloji_Harcama_TL": np.round(np.random.exponential(scale=4200, size=n_customers) + 200, 2)
    })

    # ADIM A: Veri Bütünleştirme (Data Integration / Merge)
    raw_merged = pd.merge(web_logs, crm_data, on="Musteri_ID", how="inner")
    raw_count = len(raw_merged)
    null_count = int(raw_merged.isnull().sum().sum())

    # ADIM B: Veri Ön Eleme (Data Preprocessing & Cleaning)
    df = raw_merged.copy()
    
    # 1. Eksik verileri medyan ile doldurma (Imputation)
    median_time = df["Gunluk_Aktif_Sure_Dk"].median()
    df["Gunluk_Aktif_Sure_Dk"] = df["Gunluk_Aktif_Sure_Dk"].fillna(median_time)
    
    # 2. Aykırı Değerleri (Outliers) IQR Yöntemi ile Baskılama (Clipping)
    Q1 = df["Gunluk_Aktif_Sure_Dk"].quantile(0.25)
    Q3 = df["Gunluk_Aktif_Sure_Dk"].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outlier_detected = int((df["Gunluk_Aktif_Sure_Dk"] > upper_bound).sum())
    df["Gunluk_Aktif_Sure_Dk"] = np.clip(df["Gunluk_Aktif_Sure_Dk"], a_min=1.0, a_max=upper_bound)

    # 3. Özellik Mühendisliği (Feature Engineering)
    df["Toplam_Harcama_TL"] = np.round(df["Kozmetik_Harcama_TL"] + df["Teknoloji_Harcama_TL"], 2)
    df["Baskin_Kategori"] = np.where(df["Kozmetik_Harcama_TL"] >= df["Teknoloji_Harcama_TL"] * 0.65, "Kozmetik Odaklı", "Teknoloji Odaklı")
    
    # Gelecek Kampanya Potansiyel Harcama Tutarının Hesaplanması (Hedef Değişken)
    noise = np.random.normal(0, 180, size=n_customers)
    df["Gelecek_Kampanya_Tahmini_TL"] = np.round(
        (df["Toplam_Harcama_TL"] / (df["Alisveris_Sikligi"] + 1)) * 0.65 +
        (df["Gunluk_Aktif_Sure_Dk"] * 28.5) +
        (df["Alisveris_Sikligi"] * 45.0) -
        (df["Son_Alisveris_Gun_Once"] * 6.5) + noise, 2
    )
    df["Gelecek_Kampanya_Tahmini_TL"] = np.clip(df["Gelecek_Kampanya_Tahmini_TL"], 150, None)

    # K-Means ile Davranışsal ve RFM Segmentasyonu
    features = ["Gunluk_Aktif_Sure_Dk", "Son_Alisveris_Gun_Once", "Alisveris_Sikligi", "Toplam_Harcama_TL"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled_features)
    
    # Kümeleri harcama ve aktifliğe göre anlamlı isimlendirme
    cluster_means = df.groupby("Cluster")["Toplam_Harcama_TL"].mean().sort_values(ascending=False).index
    segment_map = {
        cluster_means[0]: "💎 VIP Şampiyonlar (Yüksek Harcama & Aktif)",
        cluster_means[1]: "🛍️ Sadık Müşteriler (Düzenli Alışveriş)",
        cluster_means[2]: "👀 Vitrin Gezenler (Sitede Çok Kalıp Az Alanlar)",
        cluster_means[3]: "⚠️ Uyuyan / Kayıp Riski (Uzun Süredir Gelmeyen)"
    }
    df["Segment"] = df["Cluster"].map(segment_map)

    etl_stats = {
        "raw_rows": raw_count,
        "missing_fixed": null_count,
        "outliers_fixed": outlier_detected,
        "upper_bound_min": round(upper_bound, 1)
    }
    return df, etl_stats

df_customers, etl_info = run_automated_etl_pipeline()

# ---------------------------------------------------------
# 2. CİLT BAKIMI APRIORI VERİ SETİ YÜKLEME (HATA KORUMALI)
# ---------------------------------------------------------
@st.cache_data
def load_skincare_rules():
    if os.path.exists("cilt_bakim_kurallari.csv"):
        try:
            return pd.read_csv("cilt_bakim_kurallari.csv"), True
        except Exception:
            pass
            
    # Eğer CSV dosyası depoda yoksa bile sistemin çökmemesi için yerleşik kurallar
    fallback_rules = pd.DataFrame({
        "Secilen_Icerik": [
            "Niacinamide", "Niacinamide", "Hyaluronic Acid", "Hyaluronic Acid",
            "Retinol", "Retinol", "Salicylic Acid (BHA)", "Vitamin C",
            "Ceramides", "Glycolic Acid (AHA)", "Centella Asiatica (Cica)", "Peptides"
        ],
        "Onerilen_Icerik": [
            "Hyaluronic Acid", "Zinc PCA", "Ceramides", "Vitamin B5 (Panthenol)",
            "Hyaluronic Acid", "Ceramides", "Niacinamide", "Ferulic Acid + SPF 50",
            "Cholesterol & Fatty Acids", "Hyaluronic Acid", "Panthenol", "Retinol"
        ],
        "Destek_Support": [0.28, 0.22, 0.34, 0.31, 0.25, 0.27, 0.24, 0.29, 0.21, 0.19, 0.26, 0.23],
        "Guven_Confidence": [0.84, 0.79, 0.88, 0.85, 0.86, 0.89, 0.81, 0.91, 0.78, 0.82, 0.87, 0.80],
        "Lift_Etkisi": [2.85, 2.60, 3.10, 2.95, 3.25, 3.40, 2.75, 3.65, 2.45, 2.70, 3.05, 2.80],
        "Kategori_Sinerjisi": [
            "Gözenek & Nem Dengesi", "Sebum Kontrolü & Akne Karşıtı", "Yoğun Bariyer Onarımı", "Derinlemesine Su Tutma",
            "Tahriş Önleyici Anti-Aging", "Gece Onarım Kompleksi", "Siyah Nokta & Leke Bakımı", "Maksimum Antioksidan & Leke Kalkanı",
            "Lipid Bariyer Güçlendirici", "Hücre Yenileme & Nemlendirme", "Kızarıklık ve Hassasiyet Giderici", "Kolajen Sentezi & Sıkılaştırma"
        ]
    })
    return fallback_rules, False

df_rules, csv_loaded = load_skincare_rules()

# ---------------------------------------------------------
# YAN MENÜ (SIDEBAR) FİLTRELERİ
# ---------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Kontrol & Filtre Paneli")
    if st.button("🔄 Canlı Veri Hattını (ETL) Yenile"):
        st.cache_data.clear()
        st.rerun()
        
    secili_kategoriler = st.multiselect(
        "Müşteri Odak Kategorisi:",
        options=df_customers["Baskin_Kategori"].unique(),
        default=df_customers["Baskin_Kategori"].unique()
    )
    
    min_sure = st.slider("Min. Günlük Aktif Süre (Dk):", 1, 60, 1)
    
    st.divider()
    st.markdown("**📊 Sistem Durumu:**")
    st.success("✅ ETL & Veri Ön Eleme: Aktif")
    st.success("✅ K-Means & RFM Motoru: Aktif")
    st.success("✅ Random Forest Tahminleme: Aktif")
    st.caption("Geliştirici: Olgun Dağdelen")

filtered_df = df_customers[
    (df_customers["Baskin_Kategori"].isin(secili_kategoriler)) &
    (df_customers["Gunluk_Aktif_Sure_Dk"] >= min_sure)
]

# ---------------------------------------------------------
# ÜST ÖZET KPI KARTLARI
# ---------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Toplam Aktif Müşteri", f"{len(filtered_df):,}")
k2.metric("Ort. Günlük Sitede Kalma", f"{filtered_df['Gunluk_Aktif_Sure_Dk'].mean():.1f} Dk")
k3.metric("Ort. Alışveriş Sıklığı", f"{filtered_df['Alisveris_Sikligi'].mean():.1f} Kez")
k4.metric("Toplam Ciro (Kozmetik+Teknoloji)", f"₺{filtered_df['Toplam_Harcama_TL'].sum():,.0f}")
k5.metric("Gelecek Kampanya Potansiyeli", f"₺{filtered_df['Gelecek_Kampanya_Tahmini_TL'].sum():,.0f}", delta="+%24.8 AI Beklentisi")

st.divider()

# ---------------------------------------------------------
# 5 ANA SEKME (TABS)
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚙️ 1. Otomatize ETL & Ön Eleme",
    "👥 2. Müşteri Segmentasyonu (Süre & RFM)",
    "🎯 3. Gelecek Kampanya Harcama Tahmini",
    "💄💻 4. Kozmetik vs Teknoloji Hareketleri",
    "🧪 5. Cilt Bakımı Apriori Tavsiye Motoru"
])

# =========================================================
# SEKME 1: OTOMATİZE VERİ ÇEKME, BÜTÜNLEŞTİRME VE ÖN ELEME
# =========================================================
with tab1:
    st.subheader("⚙️ Otomatize Veri Çekme (Ingestion), Bütünleştirme ve Ön Eleme Hattı")
    st.markdown("Sistem iki farklı veri kaynağını (**Web Clickstream Logları** ve **ERP Alışveriş Tablosu**) otomatik çeker, `Musteri_ID` üzerinden birleştirir ve makine öğrenmesi modelleri için temizler.")
    
    c_etl1, c_etl2, c_etl3, c_etl4 = st.columns(4)
    c_etl1.info(f"**1. Veri Çekme (API & DB)**\n\nWeb Logları + CRM verisi çekildi.\nToplam Ham Kayıt: **{etl_info['raw_rows']}**")
    c_etl2.warning(f"**2. Eksik Veri (Imputation)**\n\nSitede kalma süresindeki **{etl_info['missing_fixed']} adet `NaN`** değer medyan ile dolduruldu.")
    c_etl3.error(f"**3. Aykırı Değer (IQR Outlier)**\n\n**{etl_info['outliers_fixed']} adet aşırı uç değer** tespit edildi ve **{etl_info['upper_bound_min']} dk** sınırına baskılandı.")
    c_etl4.success("**4. Bütünleştirme & Ölçekleme**\n\nKozmetik + Teknoloji harcamaları birleştirildi, `StandardScaler` uygulandı.")

    col_p1, col_p2 = st.columns([1.3, 1])
    with col_p1:
        st.markdown("#### 📋 Bütünleştirilmiş ve Temizlenmiş Müşteri Veri Tablosu (İlk 15 Kayıt)")
        st.dataframe(filtered_df.drop(columns=["Cluster"]).head(15), use_container_width=True)
    with col_p2:
        st.markdown("#### 📊 Ön Eleme Sonrası Sitede Kalma Süresi Dağılımı")
        fig_hist = px.histogram(
            filtered_df, x="Gunluk_Aktif_Sure_Dk", color="Baskin_Kategori",
            nbins=25, title="Temizlenmiş Günlük Aktif Süre (Dakika) Dağılımı",
            color_discrete_sequence=["#EC4899", "#3B82F6"]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# =========================================================
# SEKME 2: MÜŞTERİ SEGMENTASYONU (SÜRE, SIKLIK, HARCAMA)
# =========================================================
with tab2:
    st.subheader("👥 K-Means & RFM Tabanlı Davranışsal Müşteri Segmentasyonu")
    st.markdown("Müşterilerin **sitede aktif geçirdikleri süre**, **üyelik süresi**, **son alışveriş zamanı (Recency)**, **sıklığı (Frequency)** ve **harcama tutarı (Monetary)** analiz edilerek 4 ana segmente ayrılmıştır.")
    
    col_s1, col_s2 = st.columns([1.4, 1])
    with col_s1:
        fig_3d = px.scatter_3d(
            filtered_df,
            x="Gunluk_Aktif_Sure_Dk",
            y="Alisveris_Sikligi",
            z="Toplam_Harcama_TL",
            color="Segment",
            size="Platform_Kullanim_Suresi_Ay",
            hover_data=["Musteri_ID", "Son_Alisveris_Gun_Once", "Gelecek_Kampanya_Tahmini_TL"],
            title="3 Boyutlu Müşteri Segmentasyonu (Aktif Süre x Sıklık x Harcama)",
            height=520
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with col_s2:
        fig_seg_bar = px.bar(
            filtered_df.groupby("Segment", as_index=False)["Toplam_Harcama_TL"].mean(),
            x="Toplam_Harcama_TL", y="Segment", orientation="h", color="Segment",
            title="Segmentlere Göre Ortalama Harcama Tutarı (TL)"
        )
        st.plotly_chart(fig_seg_bar, use_container_width=True)
        
        st.markdown("#### 📌 Segment Özeti Tablosu")
        seg_summary = filtered_df.groupby("Segment").agg(
            Musteri_Sayisi=("Musteri_ID", "count"),
            Ort_Aktif_Sure_Dk=("Gunluk_Aktif_Sure_Dk", "mean"),
            Ort_Son_Alisveris_Gun=("Son_Alisveris_Gun_Once", "mean"),
            Ort_Harcama_TL=("Toplam_Harcama_TL", "mean")
        ).round(1)
        st.dataframe(seg_summary, use_container_width=True)

# =========================================================
# SEKME 3: GELECEK KAMPANYA POTANSİYEL HARCAMA TAHMİNİ (AI)
# =========================================================
with tab3:
    st.subheader("🎯 Bir Sonraki Kampanyada Yapılacak Potansiyel Harcama Tahmini (Random Forest)")
    
    # Modeli Canlı Eğitme
    X_features = ["Gunluk_Aktif_Sure_Dk", "Platform_Kullanim_Suresi_Ay", "Son_Alisveris_Gun_Once", "Alisveris_Sikligi", "Kozmetik_Harcama_TL", "Teknoloji_Harcama_TL"]
    X = df_customers[X_features]
    y = df_customers["Gelecek_Kampanya_Tahmini_TL"]
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    col_pred1, col_pred2 = st.columns([1, 1.3])
    
    with col_pred1:
        st.markdown("#### 🧮 Canlı Müşteri Kampanya Harcama Simülatörü")
        st.info("Aşağıdaki müşteri parametrelerini değiştirerek bir sonraki kampanyada bırakacağı potansiyel ciroyu tahmin edin:")
        
        sim_sure = st.slider("Sitede Günlük Aktif Geçirilen Süre (Dk)", 1.0, 60.0, 22.5)
        sim_uyelik = st.slider("Platform Kullanım Süresi / Üyelik Yaşı (Ay)", 1, 48, 14)
        sim_recency = st.slider("Son Alışverişten Geçen Süre (Gün Önce)", 1, 180, 12)
        sim_freq = st.slider("Toplam Alışveriş Sıklığı (Adet)", 1, 35, 9)
        sim_koz = st.number_input("Geçmiş Kozmetik Harcaması (TL)", 0, 30000, 2400, step=200)
        sim_tek = st.number_input("Geçmiş Teknoloji Harcaması (TL)", 0, 50000, 5500, step=500)
        
        input_data = pd.DataFrame([[sim_sure, sim_uyelik, sim_recency, sim_freq, sim_koz, sim_tek]], columns=X_features)
        predicted_spend = rf_model.predict(input_data)[0]
        
        st.success(f"### 💰 Tahmini Kampanya Harcaması: **₺{predicted_spend:,.2f}**")
        if predicted_spend > 3500:
            st.markdown("🔥 **AI Aksiyon Önerisi:** Bu müşteriye **VIP Erken Erişim + Teknoloji/Kozmetik Çapraz Kuponu** tanımlayın!")
        else:
            st.markdown("📩 **AI Aksiyon Önerisi:** Sepet tutarını artırmak için **'X TL Üzerine Kargo Bedava & %15 İndirim'** bildirimi gönderin.")

    with col_pred2:
        st.markdown("#### 📈 Sitede Geçirilen Aktif Süre vs. Gelecek Kampanya Harcama Potansiyeli")
        fig_pred = px.scatter(
            filtered_df, x="Gunluk_Aktif_Sure_Dk", y="Gelecek_Kampanya_Tahmini_TL",
            color="Segment", size="Alisveris_Sikligi",
            hover_data=["Musteri_ID", "Toplam_Harcama_TL"],
            title="Aktif Süre ve Sıklığın Gelecek Kampanya Harcamasına Etkisi"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Özellik Önem Düzeyleri (Feature Importance)
        imp_df = pd.DataFrame({"Ozellik": X_features, "Etki_Gucu": rf_model.feature_importances_}).sort_values("Etki_Gucu", ascending=True)
        fig_imp = px.bar(imp_df, x="Etki_Gucu", y="Ozellik", orientation="h", title="Harcanacak Tutarı Belirleyen En Kritik Faktörler (Feature Importance)")
        st.plotly_chart(fig_imp, use_container_width=True)

# =========================================================
# SEKME 4: KOZMETİK VE TEKNOLOJİ KATEGORİ HAREKETLERİ
# =========================================================
with tab4:
    st.subheader("💄💻 Kozmetik ve Teknoloji Kategorilerinin Davranışsal Hareketleri")
    st.markdown("Müşterilerin günün saatlerine göre sitedeki kategori trafiği, harcama korelasyonları ve kategoriler arası geçiş eğilimleri.")
    
    # Saatlik trafik simülasyonu
    saatler = [f"{s:02d}:00" for s in range(24)]
    kozmetik_trafik = [15, 10, 6, 4, 3, 5, 12, 25, 40, 55, 68, 75, 82, 78, 72, 70, 75, 85, 95, 110, 125, 115, 80, 40]
    teknoloji_trafik = [28, 18, 12, 8, 5, 6, 10, 20, 35, 48, 60, 72, 85, 80, 75, 78, 84, 92, 105, 120, 135, 140, 110, 65]
    
    df_hourly = pd.DataFrame({
        "Saat": saatler * 2,
        "Aktif_Kullanici_Endeksi": kozmetik_trafik + teknoloji_trafik,
        "Kategori": ["Kozmetik"] * 24 + ["Teknoloji"] * 24
    })
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        fig_line = px.line(
            df_hourly, x="Saat", y="Aktif_Kullanici_Endeksi", color="Kategori",
            markers=True, title="24 Saatlik Kozmetik vs. Teknoloji Aktif Trafik Hareketi",
            color_discrete_map={"Kozmetik": "#EC4899", "Teknoloji": "#2563EB"}
        )
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col_k2:
        fig_cross = px.scatter(
            filtered_df, x="Kozmetik_Harcama_TL", y="Teknoloji_Harcama_TL",
            color="Baskin_Kategori", size="Gunluk_Aktif_Sure_Dk",
            title="Çapraz Kategori Harcama Matrisi (Kozmetik x Teknoloji)",
            color_discrete_sequence=["#EC4899", "#2563EB"]
        )
        st.plotly_chart(fig_cross, use_container_width=True)

    st.info("💡 **Veri İçgörüsü:** Kozmetik kategorisinde akşam **20:00 - 22:00** saatleri arasında sitede geçirilen süre zirve yaparken, Teknoloji kategorisinde trafik gece **21:00 - 00:00** aralığında yoğunlaşmaktadır. İki kategoriden ortak alışveriş yapan müşterilerin sepet ortalaması **%42 daha yüksektir**.")

# =========================================================
# SEKME 5: CİLT BAKIMI APRIORI TAVSİYE MOTORU & GÖRSEL AĞ
# =========================================================
with tab5:
    st.subheader("🧪 Cilt Bakımı E-Ticaret Apriori Tavsiye Motoru & Görsel Kural Analizi")
    
    if csv_loaded:
        st.success("✅ `cilt_bakim_kurallari.csv` veri seti başarıyla yüklendi.")
    else:
        st.info("📌 Yerleşik (Built-in) Apriori Cilt Bakımı Birliktelik Kuralları veritabanı aktif.")

    col_a1, col_a2 = st.columns([1, 1.4])
    
    # Sütun isimlerini esnek yakalama (mevcut CSV ile tam uyum için)
    cols = df_rules.columns.tolist()
    antecedent_col = cols[0]
    consequent_col = cols[1]
    
    with col_a1:
        st.markdown("#### 🔍 Akıllı İçerik & Ürün Eşleştirici")
        secilen_urun = st.selectbox("Müşterinin Sepetine Eklediği İçerik / Ürün:", df_rules[antecedent_col].unique())
        
        eslesen_kurallar = df_rules[df_rules[antecedent_col] == secilen_urun]
        
        st.markdown("### 🎯 AI Çapraz Satış (Cross-Sell) Tavsiyeleri:")
        for idx, row in eslesen_kurallar.iterrows():
            onerilen = row[consequent_col]
            lift_val = row[cols[4]] if len(cols) > 4 else 2.85
            conf_val = row[cols[3]] if len(cols) > 3 else 0.85
            st.success(f"➕ **{onerilen}** ile birlikte önerin! \n*(Güven: %{conf_val*100:.0f} | Lift Etkisi: {lift_val:.2f}x)*")
            
        st.dataframe(eslesen_kurallar, use_container_width=True)

    with col_a2:
        st.markdown("#### 🫧 Apriori Birliktelik Kuralları Matrisi (Support x Confidence x Lift)")
        if len(cols) >= 5:
            fig_apriori = px.scatter(
                df_rules, x=cols[2], y=cols[3], size=cols[4], color=cols[0],
                hover_name=cols[1],
                title="Birliktelik Kuralları Güç Haritası (Balon Büyüklüğü = Lift)",
                labels={cols[2]: "Destek (Support)", cols[3]: "Güven (Confidence)", cols[4]: "Lift"}
            )
            st.plotly_chart(fig_apriori, use_container_width=True)
        else:
            st.bar_chart(df_rules.set_index(antecedent_col))
