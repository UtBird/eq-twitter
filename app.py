import streamlit as st
import json
import folium
from streamlit_folium import st_folium
import sys
import os

# Add the current working directory to the system path to allow importing src
sys.path.append(os.getcwd())

from src.pipeline import DisasterPipeline

# -- Sayfa Ayarları --
st.set_page_config(
    page_title="P-5 Afet NLP Analiz Ağı",
    page_icon="🚨",
    layout="wide"
)

# Modelleri sadece ilk yüklemede belleğe almak için cache (Resource Cache)
@st.cache_resource
def load_pipeline():
    with st.spinner("NLP ve Multimodal modeller (BERTurk, NER, Zeyrek) yükleniyor. Bu işlem bir kez yapılır ve yaklaşık 30 saniye sürebilir..."):
        return DisasterPipeline()

st.title("🚨 P-5: Afet Metin & Multimodal Veri Füzyonu GUI")
st.markdown("""
Bu arayüz, sosyal medyadan elde edilen kentsel afet verisinin uçtan uca analizini simüle etmektedir.
1. **Zeyrek** ile metni sadeleştirir.
2. **BERTurk Sınıflandırma** ile P-5 kategorilerini (Yol/Enkaz) belirler ve güven skoru ölçer.
3. **BERTurk NER** ile metinden adres bilgisini çeker.
4. **GeoPy** ile çekilen adresi harita koordinatlarına dönüştürür.
5. Veriyi **Düşük Bant Genişliğine (JSON)** optimize eder.
""")

pipeline = load_pipeline()

# -- Kullanıcı Arayüzü --
st.subheader("Simülasyon Verisi Gönder")

# Hazır bazı örnek tweetler
sample_texts = [
    "Hatay antakya cebrail mahallesi yıkıldı, enkaz altında kalanlar var lütfen yardım edin ses geliyor!",
    "Gaziantep nurdağı yolu kapalı tırlar geçemiyor, toprak kayması var.",
    "Kahramanmaraş merkezde 50 çadır ve bol miktarda bebek maması ihtiyacı çok acil."
]

selected_sample = st.selectbox("Örnek Test Verisi Seçin", ["Lütfen kendi metnini yazını kullanın..."] + sample_texts)

user_input = st.text_area(
    "Veya Sosyal medya (X) metni girin:", 
    value=selected_sample if selected_sample != "Lütfen kendi metnini yazını kullanın..." else ""
)

if st.button("Uçtan Uca Analizi Çalıştır", type="primary"):
    if not user_input.strip():
        st.warning("Lütfen analiz edilecek bir metin girin.")
    else:
        with st.spinner("Metin işleniyor, konum çıkartılıyor... (Multimodal NLP Pipeline Devrede)"):
            result = pipeline.process_tweet(user_input)
            
        if not result:
            st.error("Bu girdi, afet yönetim çerçevesine uymadığı için (Alakasız) reddedildi.")
        else:
            # Düzen (Sol: Pano, Sağ: JSON)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.success("✅ Veri İşleme Başarılı!")
                st.metric("Tespit Edilen Kategori", result["kategori"])
                st.metric("Model Güven Skoru", f"%{result['guven_skoru'] * 100:.1f}")
                
                aciliyet = result["aciliyet"]
                st.metric("P-5 Aciliyet Seviyesi (1-5)", aciliyet)
                
                if aciliyet >= 4:
                    st.error("⚠️ KRİTİK ACİLİYET DURUMU. Önceliklendirme Gereklidir.")
                elif aciliyet == 3:
                    st.warning("🚧 Rota Bildirimi. Lojistik ve Ulaşım Algoritmaları Tetiklenmelidir.")
                    
            with col2:
                st.markdown("### 📡 Düşük Bant Genişliği İletim Formatı (P-5 JSON V1)")
                st.code(json.dumps(result, ensure_ascii=False, indent=4), language="json")

            # Harita Bölümü
            st.markdown("### 🌍 Varlık Çıkarımı (NER) ve Uzamsal Haritalama")
            coords = result.get("konum")
            if coords:
                lat, lon = coords
                st.info(f"Metin içerisinden otonom olarak konum çıkartıldı ve Geocoding çalıştırıldı: Enlem: \u200e{lat}, Boylam: \u200e{lon}")
                m = folium.Map(location=coords, zoom_start=15)
                folium.Marker(
                    coords, popup=result["kategori"], tooltip="Tespit Edilen Konum"
                ).add_to(m)
                st_folium(m, height=400, use_container_width=True)
            else:
                st.info("Bu metin içerisinde açık bir konum bilgisine rastlanmadı veya algoritmalar spesifik bir koordinata çözümleyemedi.")

st.sidebar.markdown("### Mimari Bileşenler")
st.sidebar.caption("- Zemberek NLP Modülü (Zeyrek)")
st.sidebar.caption("- HuggingFace: deprem-ml/multilabel_earthquake_tweet_intent_bert_base_turkish_cased")
st.sidebar.caption("- HuggingFace: yhaslan/turkish-earthquake-tweets-ner")
st.sidebar.caption("- GeoPy & Folium (Harita)")

