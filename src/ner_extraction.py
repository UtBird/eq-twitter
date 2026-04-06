from transformers import pipeline
import torch
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

class LocationExtractor:
    def __init__(self, model_name="yhaslan/turkish-earthquake-tweets-ner"):
        device = 0 if torch.cuda.is_available() else -1
        # Aggregation strategy groups tokens into full entities (e.g., 'Avcılar' 'Merkez' -> 'Avcılar Merkez')
        self.ner = pipeline("token-classification", model=model_name, aggregation_strategy="simple", device=device)
        
        # Geopy setup with OpenStreetMap
        self.geolocator = Nominatim(user_agent="disaster_nlp_app")
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1.2)
        
    def extract_entities(self, text):
        """
        Extracts named entities from the text (Location, Organization, etc.)
        """
        if not text:
            return []
            
        try:
            return self.ner(text[:512])
        except:
            return []

    def get_coordinates(self, location_text):
        """
        Converts a location string (e.g., "Hatay Antakya") to GPS coordinates.
        Uses progressive fallback if the exact street query fails.
        """
        words = location_text.split()
        while len(words) > 0:
            query = " ".join(words) + ", Turkey"
            try:
                location = self.geocode(query)
                if location:
                    return [location.latitude, location.longitude]
            except Exception:
                pass
            # Remove the last word and try again (e.g. "Hatay Belen geçidi" -> "Hatay Belen")
            words.pop()
            
        return None

if __name__ == "__main__":
    extractor = LocationExtractor()
    sample = "Hatay antakya cumhuriyet caddesi bina yıkıldı acil yardım"
    entities = extractor.extract_entities(sample)
    print("Entities:", entities)
    
    # Extract just the locations (if model labels them as LOC)
    locs = [ent['word'] for ent in entities if 'LOC' in ent['entity_group']]
    if locs:
        full_loc = " ".join(locs)
        coords = extractor.get_coordinates(full_loc)
        print(f"Address: {full_loc} -> Coords: {coords}")
