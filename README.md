# ⚡ NEXUS AI: Enterprise E-Commerce Intelligence & Decision Suite (v7.0)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%2520Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%2520Analytics-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 💎 Proje Hakkında (About)
**NEXUS AI**, e-ticaret operasyonlarını uçtan uca optimize etmek amacıyla geliştirilmiş, yapay zeka destekli **Kurumsal Karar Destek ve Veri Mühendisliği Platformudur**. Statik veri setleri yerine dış kaynaklardan ve **Trendyol** alt yapısından otonom veri çeken, gelişmiş veri ön eleme (ETL) adımlarından geçiren ve çok katmanlı makine öğrenmesi modelleri koşturan hibrit bir ekosistemdir.

---

## 🏗️ 12 Devasa Kurumsal Modül & Özellik Seti

1. **🕷️ Canlı Web Scraping & ETL Hattı:** Python `requests` ve `BeautifulSoup` ile e-ticaret sunucularından canlı SKU'ları çeker; medyan ataması (`Imputation`) ve **IQR (Çeyrekler Açıklığı)** yöntemiyle aykırı değerleri (`outliers`) temizler.
2. **📊 Gelişmiş Görsel Veri Keşfi (EDA):** Pearson Korelasyon Isı Haritası (`Heatmap`), Şehir Bazlı Kemancı Grafikleri (`Violin Plot`) ve Türkiye Hiyerarşik Ciro Haritaları (`Treemap`).
3. **👥 3 Boyutlu RFM & Radar Segmentasyonu:** Müşterileri site içi aktif süre, sıklık ve harcama metriklerine göre `K-Means` ile kümeleyip 3B uzayda ve 5 boyutlu Radar (Spider) grafiğinde analiz eder.
4. **🎯 Kampanya Ciro Tahmini:** `Random Forest Regressor` modeli ile müşterilerin gelecek kampanyalarda bırakacağı potansiyel ciroyu (TL) tahmin eden interaktif simülatör.
5. **📈 Teknoloji & Cilt Bakımı Çapraz Regresyonu:** İki kategori arasındaki doğrusal/polinomiyal regresyon denklemlerini ($Y = \beta_0 + \beta_1 X$), Pearson korelasyonunu ($r$) ve $R^2$ açıklayıcılık oranını matematiksel olarak modeller.
6. **🧪 Apriori Tavsiye Motoru:** Cilt bakımı bileşenleri (*Niacinamide, Retinol, Hyaluronic Acid vb.*) arasındaki kimyasal ve ticari sinerjiyi **Support, Confidence ve Lift** metrikleriyle çözer.
7. **💬 NLP Duygu Analizi & Sahte Bot Avcısı:** Müşteri yorumlarını doğal dil işleme ile tarar; `Isolation Forest` denetimsiz anomali tespitiyle şüpheli bot/manipülasyon saldırılarını saptar.
8. **💸 AI Dinamik Fiyatlandırma:** Fiyat elastikiyeti formülüyle indirim/zam oranlarının net kâra ve satış adetlerine etkisini hesaplayıp maksimum kâr noktasını bulur.
9. **📦 Akıllı Stok & Tedarik Radarı:** Ürünlerin günlük erime hızına (`Burn Rate`) göre kaç gün sonra tükeneceğini hesaplayıp kritik stok alarmları üretir.
10. **⚖️ A/B Kampanya Hipotez Testi:** Pazarlama kurgularının dönüşüm oranlarını Z-Testi ve p-value istatistiksel anlamlılık motoruyla test eder.
11. **⚔️ BuyBox Rakip Fiyat İstihbaratı:** Trendyol, Hepsiburada ve Amazon Türkiye fiyatlarını kıyaslayıp en avantajlı satıcı (BuyBox) stratejisini önerir.
12. **🤖 AI Reçete Danışmanı & Kohort (Cohort) Retention:** Kullanıcı bütçesine özel sepet reçetesi üretir ve müşteri elde tutma oranlarını üçgen ısı haritasıyla gösterir.
13. **🟠 Otonom Trendyol Canlı Scraper:** Trendyol DOM yapısını (`div.p-card-wrppr`) tarayan ve hibrit önbellek akışıyla çalışan otonom fiyat takip daemon'u.

---

## 🛠️ Kullanılan Teknolojiler & Kütüphaneler
* **Programming Language:** Python 3.10+
* **Web Framework:** Streamlit Cloud
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means, Random Forest, Linear Regression, Isolation Forest)
* **Web Scraping:** Requests, BeautifulSoup4
* **Data Visualization:** Plotly (Express & Graph Objects)

---

## 🚀 Yerel Çalıştırma (Quick Start)

Depoyu klonlayın ve bağımlılıkları yükleyin:

```bash
git clone [https://github.com/KULLANICI_ADINIZ/cilt-bakim-apriori.git](https://github.com/KULLANICI_ADINIZ/cilt-bakim-apriori.git)
cd cilt-bakim-apriori
pip install -r requirements.txt
streamlit run app.py
