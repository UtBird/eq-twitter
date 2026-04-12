# EQ Twitter Streamlit App

Bu proje, deprem odakli sosyal medya metinlerini siniflandirip konum cikarmak icin hazirlanmis bir Streamlit uygulamasidir.

## Kurulum

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model Ayari

Siniflandirma modeli GitHub reposuna dahil edilmez. Uygulamayi calistirmadan once Hugging Face model adini ortam degiskeni olarak ver:

```bash
export DISASTER_MODEL_NAME="BURAYA_HUGGINGFACE_MODEL_ADINI_YAZ"
```

## Calistirma

```bash
streamlit run app.py
```

Ilk calistirmada NER modeli Hugging Face uzerinden indirilebilir. Siniflandirma modeli de `DISASTER_MODEL_NAME` icindeki Hugging Face repo adindan yuklenir.

Hugging Face model linki:

- Buraya kendi model linkini ekle

## Dataset Kaynagi

Projede kullanilan deprem tweet datasetinin kaynak linki:

- Kaggle: https://www.kaggle.com/datasets/ulkutuncerkucuktas/turkey-earthquake-relief-tweets-dataset
