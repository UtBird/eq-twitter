import torch
from transformers import pipeline

# P-5 Required Categories
# enkaz bildirimi, yol kapanma bilgisi, acil yardım ihtiyacı, lojistik ihtiyaç bildirimi
CATEGORY_MAPPING = {
    "Kurtarma": "Enkaz Bildirimi",
    "Saglik": "Acil Yardım İhtiyacı",
    "Su": "Acil Yardım İhtiyacı",
    "Yemek": "Acil Yardım İhtiyacı",
    "Barinma": "Lojistik İhtiyaç Bildirimi",
    "Giysi": "Lojistik İhtiyaç Bildirimi",
    "Lojistik": "Yol Kapanma Bilgisi / Lojistik", 
    "Elektronik": "Lojistik İhtiyaç Bildirimi",
    "Alakasiz": "Alakasız"
}

class DisasterClassifier:
    def __init__(self, model_name="deprem-ml/multilabel_earthquake_tweet_intent_bert_base_turkish_cased"):
        device = 0 if torch.cuda.is_available() else -1
        # The model is multilabel, so top_k=None returns all scores
        self.classifier = pipeline("text-classification", model=model_name, device=device, top_k=None)
        
    def classify(self, text):
        """
        Classifies the text and maps it to P-5 standard categories.
        Returns the top category and its confidence score.
        """
        # Protect against empty strings
        if not text or len(text.strip()) < 5:
            return "Alakasız", 0.0

        try:
            predictions = self.classifier(text[:512])[0]
            
            # predictions is a list of dicts like [{'label': 'Kurtarma', 'score': 0.9}, ...]
            # Sort by score descending
            predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
            
            top_pred = predictions[0]
            raw_label = top_pred['label']
            score = top_pred['score']
            
            # Map to P-5 specific label
            p5_label = CATEGORY_MAPPING.get(raw_label, "Diğer")
            
            # If the highest score is very low, or it's "Alakasiz", return Alakasiz
            if score < 0.3 or raw_label == "Alakasiz":
                 return "Alakasız", score

            # Fallback heuristics for custom subcategories
            if raw_label == "Lojistik" and ("yol" in text.lower() or "kapalı" in text.lower()):
                p5_label = "Yol Kapanma Bilgisi"
            elif raw_label == "Lojistik":
                p5_label = "Lojistik İhtiyaç Bildirimi"

            return p5_label, score

        except Exception as e:
            return "Hata", 0.0

if __name__ == "__main__":
    clf = DisasterClassifier()
    sample = "Avcılar caddesi yol çöktü, iş makinesi giremiyor acil lojistik lazım."
    cat, score = clf.classify(sample)
    print(f"Text: {sample}")
    print(f"Kategori: {cat} (Skor: {score:.4f})")
