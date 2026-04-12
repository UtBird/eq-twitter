import json
from src.preprocessing import clean_text, zemberek_lemmatize
from src.classification import DisasterClassifier
from src.ner_extraction import LocationExtractor

class DisasterPipeline:
    def __init__(self):
        print("Mimariler yükleniyor (Zeyrek, BERTurk Sınıflandırma, BERTurk NER)...")
        self.classifier = DisasterClassifier()
        self.ner_extractor = LocationExtractor()
        
    def process_tweet(self, raw_text):
        """
        Runs the end-to-end pipeline:
        1. Preprocessing (Clean & Lemmatize)
        2. Classification (Intent & Confidence)
        3. NER & Geocoding (Location Extraction)
        4. JSON Formatting (Low Bandwidth Struct)
        """
        # 1. Preprocessing
        clean_tx_cased = clean_text(raw_text, lowercase=False)  # Keep case for NER!
        clean_tx_lower = clean_text(raw_text, lowercase=True)   # For Zeyrek
        lemmatized_text = zemberek_lemmatize(clean_tx_lower)
        
        # 2. Classification
        category, confidence = self.classifier.classify(clean_tx_cased)
        
        # If it's not a relevant disaster category, we can skip geocoding to save bandwidth
        if category == "Alakasız":
            return None
            
        # 3. NER Extraction
        entities = self.ner_extractor.extract_entities(clean_tx_cased)
        
        # yhaslan/turkish-earthquake-tweets-ner uses 'ADDR' and 'CITY'
        locations = [ent['word'] for ent in entities if ent['entity_group'] in ['ADDR', 'CITY']]
        
        coords = None
        if locations:
            # Combine extracted locations to form a single query (e.g., "Hatay", "Antakya" -> "Hatay Antakya")
            full_address = " ".join(locations)
            coords = self.ner_extractor.get_coordinates(full_address, raw_text=clean_tx_cased)
            
        # Derive urgency level (1-5) based on category and confidence
        aciliyet = 1
        if category == "Enkaz Bildirimi":
            aciliyet = 5
        elif category == "Acil Yardım İhtiyacı":
            aciliyet = 4
        elif category == "Yol Kapanma Bilgisi / Lojistik":
            aciliyet = 3
        elif category == "Lojistik İhtiyaç Bildirimi":
            aciliyet = 2

        # 4. Low Bandwidth Output Structure JSON
        output = {
            "kategori": category,
            "konum": coords,
            "aciliyet": aciliyet,
            "guven_skoru": float(round(confidence, 3))
        }
        
        return output

if __name__ == "__main__":
    pipeline = DisasterPipeline()
    
    test_tweets = [
        "Hatay antakya cebrail mahallesi yıkıldı, enkaz altında kalanlar var lütfen yardım edin ses geliyor!",
        "Gaziantep nurdağı yolu kapalı tırlar geçemiyor.",
        "Malatya merkezde çadır ve battaniye ihtiyacı çok acil."
    ]
    
    print("\n--- TEST BAŞLIYOR ---\n")
    for tw in test_tweets:
        print(f"Orijinal: {tw}")
        result = pipeline.process_tweet(tw)
        print(f"Çıktı JSON: {json.dumps(result, ensure_ascii=False, indent=2)}\n")
