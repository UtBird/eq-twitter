from src.classification import DisasterClassifier
from src.ner_extraction import LocationExtractor

clf = DisasterClassifier()
ner = LocationExtractor()

text = "Adana Pozantı otoyolu ulaşıma tamamen kapalı, dağdan kayalar düşmüş tırlar bekliyor acil kar küreme aracı veya iş makinesi lazım."

print("--- CLASSIFICATION ---")
cat, score = clf.classify(text)
print("Cat:", cat, "| Score:", score)

# Raw pipeline predictions
preds = clf.classifier(text[:512])[0]
print("All Preds:", sorted(preds, key=lambda x: x['score'], reverse=True)[:3])

print("\n--- NER ---")
entities = ner.extract_entities(text)
print("Entities:", entities)

# Filter locations
locations = [ent['word'] for ent in entities if ent['entity_group'] in ['ADDR', 'CITY']]
print("Extracted LOC Strings:", locations)

if locations:
    full_address = " ".join(locations)
    print("Full Address string to pass to GeoPy:", full_address)
    coords = ner.get_coordinates(full_address)
    print("GeoPy Coordinates:", coords)
else:
    print("No valid ADDR or CITY entities found.")
