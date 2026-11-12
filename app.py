import streamlit as st
import pandas as pd

st.set_page_config(page_title="Skincare AI Tavsiye Motoru", layout="centered")
st.title("🧪 Cilt Bakımı E-Ticaret Tavsiye Motoru")
st.markdown("Bu otomasyon, Apriori veri madenciliği modeliyle e-ticaret verilerinden öğrenilmiş kimyasal formül kurallarını kullanarak, seçtiğiniz ürün içeriğine en uygun diğer içerikleri tavsiye eder.")
st.divider()

@st.cache_data
def load_data():
    df = pd.read_csv('cilt_bakim_kurallari.csv')
    df['antecedents'] = df['antecedents'].astype(str).str.replace("frozenset", "").str.replace(r"[{()\'}]", "", regex=True)
    df['consequents'] = df['consequents'].astype(str).str.replace("frozenset", "").str.replace(r"[{()\'}]", "", regex=True)
    df = df[df['antecedents'].str.len() > 2]
    df = df[df['consequents'].str.len() > 2]
    return df

df_rules = load_data()

benzersiz_icerikler = sorted(df_rules['antecedents'].unique())
secilen_icerik = st.selectbox("🔍 Tavsiye motoru için bir ana içerik/bileşen seçin:", benzersiz_icerikler)

if st.button("🚀 Otomasyonu Çalıştır"):
    sonuclar = df_rules[df_rules['antecedents'] == secilen_icerik].copy()
    
    if not sonuclar.empty:
        st.success(f"✅ **{secilen_icerik}** içeren bir ürünü alan müşteriye tavsiye edilmesi gerekenler:")
        gosterim_tablosu = sonuclar[['consequents', 'confidence', 'lift']].copy()
        gosterim_tablosu['confidence'] = (gosterim_tablosu['confidence'] * 100).round(2).astype(str) + " %"
        gosterim_tablosu['lift'] = gosterim_tablosu['lift'].round(2)
        gosterim_tablosu.rename(columns={
            'consequents': 'Tavsiye Edilen İçerik (Sonuç)', 
            'confidence': 'Birlikte Alınma Olasılığı (Güven)', 
            'lift': 'Bağlılık Gücü (Lift)'
        }, inplace=True)
        st.dataframe(gosterim_tablosu.head(10), use_container_width=True, hide_index=True)
    else:
        st.warning("Bu içerik için yeterince güçlü bir tavsiye kuralı bulunamadı.")
