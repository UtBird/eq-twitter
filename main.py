import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from transformers import pipeline
from tqdm import tqdm
import re
import os
import torch

# 1. Konfigürasyon ve Görselleştirme Ayarları
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.family'] = 'sans-serif'

# Kritik Anahtar Kelimeler (Filtreleme için)
EMERGENCY_KEYWORDS = [
    'yardım', 'enkaz', 'acil', 'altında', 'ses', 'bina', 'adres', 'sokak', 'mahalle', 
    'apartman', 'kat', 'no', 'daire', 'rescue', 'help', 'trapped', 'emergency', 'sos'
]

def clean_text(text):
    """Tweet metnini temizler."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'RT @\S+:', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_potential_emergency(text):
    """Tweet acil durum kelimeleri içeriyor mu?"""
    text_lower = text.lower()
    return any(kw in text_lower for kw in EMERGENCY_KEYWORDS)

def classify_disaster_response(texts, model_name="yhaslan/berturk-earthquake-tweets-classification"):
    """yhaslan'ın modelini kullanarak tweetleri sınıflandırır."""
    print(f"\n--- BERTurk Acil Çağrı Taraması Başlatılıyor ({model_name}) ---")
    
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("text-classification", model=model_name, device=device)
    
    results = []
    for text in tqdm(texts, desc="Kritik tweetler analiz ediliyor"):
        if len(text.strip()) < 5:
            results.append("LABEL_0")
            continue
        try:
            prediction = classifier(text[:512])[0]
            results.append(prediction['label'])
        except Exception as e:
            results.append("LABEL_0")
    
    return results

def analyze_dataset(file_path):
    print(f"--- {file_path} TAM VERİ SETİ TARAMASI ---")
    
    # 1. Veriyi yükle
    df = pd.read_csv(file_path)
    print(f"Toplam Tweet Sayısı: {len(df)}")
    
    # 2. Ön Filtreleme (Hız İçin)
    print(f"Adım 1/3: Anahtar kelime filtresi uygulanıyor...")
    df['clean_text'] = df['text'].apply(clean_text)
    df['is_candidate'] = df['clean_text'].apply(is_potential_emergency)
    
    candidates = df[df['is_candidate']].copy()
    print(f"-> 28,844 tweet içinden {len(candidates)} potansiyel kritik tweet ayrıştırıldı.")
    
    # 3. BERTurk Modelle Sınıflandırma (Sadece Adaylar Üzerinde)
    print(f"Adım 2/3: Model {len(candidates)} tweet üzerinde çalıştırılıyor...")
    if not candidates.empty:
        class_results = classify_disaster_response(candidates['clean_text'].tolist())
        candidates['disaster_label'] = class_results
    else:
        candidates['disaster_label'] = []

    # Sonuçları Ana Veri Setine Birleştir
    df['disaster_label'] = 'LABEL_0' # Varsayılan olarak hep "Acil Değil"
    df.loc[df['is_candidate'], 'disaster_label'] = candidates['disaster_label'].values
    
    # Gerçek Acil Çağrıları (LABEL_1) Ayır ve Kaydet
    emergency_calls = df[df['disaster_label'] == 'LABEL_1'].copy()
    print(f"Adım 3/3: Filtreleme bitti. Toplam {len(emergency_calls)} GERÇEK ACİL ÇAĞRI bulundu.")
    
    if not emergency_calls.empty:
        emergency_file = "emergency_calls_found.csv"
        emergency_calls[['date', 'user_name', 'text', 'disaster_label']].to_csv(emergency_file, index=False)
        print(f"-> Kritik çağrılar '{emergency_file}' dosyasına kaydedildi. Lütfen inceleyin!")
    
    # 4. Görselleştirme
    print("Görseller oluşturuluyor...")
    
    # Acil Çağrı Dağılımı
    plt.figure()
    label_map = {'LABEL_0': 'Acil Değil/Genel Bilgi', 'LABEL_1': 'KRİTİK ACİL ÇAĞRI'}
    df['human_label'] = df['disaster_label'].map(label_map)
    label_counts = df['human_label'].value_counts()
    
    sns.barplot(x=label_counts.values, y=label_counts.index, palette=['gray', 'red'])
    plt.title('Tüm Veri Seti: Acil Çağrı Analizi', fontsize=15)
    plt.xlabel('Tweet Sayısı')
    plt.tight_layout()
    plt.savefig('disaster_classification_full.png')
    print("-> disaster_classification_full.png kaydedildi.")

    # Kelime Bulutu (Sadece Acil Çağrılar İçin)
    if not emergency_calls.empty:
        all_text = " ".join(emergency_calls['clean_text'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(all_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Acil Çağrılarda En Sık Geçen Kelimeler', fontsize=15)
        plt.tight_layout()
        plt.savefig('emergency_wordcloud.png')
        print("-> emergency_wordcloud.png kaydedildi.")

    print("\n--- ANALİZ TAMAMLANDI ---")
    print(f"Toplam {len(df)} tweet tarandı.")
    print(f"Sonuç: {len(emergency_calls)} gerçek kurtarma çağrısı tespit edildi.")

if __name__ == "__main__":
    file_name = "turkey_earthquake_tweets.csv"
    if os.path.exists(file_name):
        analyze_dataset(file_name)
    else:
        print(f"Hata: {file_name} dosyası bulunamadı!")

if __name__ == "__main__":
    file_name = "turkey_earthquake_tweets.csv"
    if os.path.exists(file_name):
        analyze_dataset(file_name)
    else:
        print(f"Hata: {file_name} dosyası bulunamadı!")
