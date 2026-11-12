# 🧪 Cilt Bakımı E-Ticaret Tavsiye Motoru (Skincare AI Recommender)

Bu proje, e-ticaret kozmetik ve cilt bakımı ürünlerinin içerik (ingredient) verilerini **Veri Madenciliği (Apriori Algoritması)** ile analiz ederek, birlikte en sık kullanılan ve birbirini tamamlayan bileşenleri tespit eden interaktif bir tavsiye otomasyonudur.

## 🌐 Canlı Web Uygulaması (Live Demo)
Projeyi tarayıcı üzerinden 7/24 canlı olarak test etmek için aşağıdaki bağlantıya tıklayabilirsiniz:
👉 **[Canlı Uygulamayı Açmak İçin Tıklayın](https://cilt-bakim-apriori-fecuugr3ztb8ahcl8twnz4.streamlit.app/)**

---

## 🎯 Projenin Amacı ve Çalışma Mantığı
E-ticaret platformlarında müşterilere doğru ürün ve içerik kombinasyonlarını önermek (Cross-Selling) amacıyla geliştirilmiştir:
1. **Veri Ön İşleme:** Kozmetik ürünlerin kimyasal içerik listeleri temizlenmiş ve Apriori modeline uygun sepet/işlem (transaction) matrisine dönüştürülmüştür.
2. **Birliktelik Kuralları (Association Rules):** `mlxtend` kütüphanesi ve **Apriori** algoritması kullanılarak içerikler arasındaki **Destek (Support)**, **Güven (Confidence)** ve **İlgi/Bağlılık (Lift)** metrikleri hesaplanmıştır.
3. **Web Otomasyonu:** Çıkarılan kurallar (`cilt_bakim_kurallari.csv`), **Streamlit** tabanlı dinamik bir arayüze entegre edilerek son kullanıcının seçtiği içeriğe göre anlık tavsiye üreten bir motora dönüştürülmüştür.

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
* **Dil:** Python 3
* **Veri Analizi ve Modelleme:** Pandas, NumPy, mlxtend (Apriori & Association Rules)
* **Web Arayüzü ve Dağıtım (Deployment):** Streamlit, Streamlit Community Cloud

## 📁 Depo İçeriği
* `app.py`: Streamlit web arayüzü ve tavsiye motoru kaynak kodu.
* `cilt_bakim_kurallari.csv`: Apriori modeli sonucunda üretilen birliktelik kuralları veri seti.
* `requirements.txt`: Bulut sunucu kurulumu için gerekli kütüphane bağımlılıkları.
