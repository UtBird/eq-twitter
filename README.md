# EQ Twitter

EQ Twitter is a Streamlit application for analyzing earthquake-related social media posts. It classifies incoming text, extracts location entities, and visualizes the detected result on a map.

## Features

- Earthquake-related text classification
- Location extraction with NER
- Coordinate lookup and map visualization
- Streamlit interface for quick testing

## Project Structure

```text
.
├── app.py
├── main.py
├── requirements.txt
├── src/
├── models/
│   └── 2kveri/
└── notebooks/
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Model Setup

The classification model is not included in this GitHub repository.

To run the app, download the model files from Hugging Face and place them inside:

```text
models/2kveri/
```

Required files:

- `config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

Hugging Face model link:

- Add your model link here

After downloading, your folder structure should look like this:

```text
models/
└── 2kveri/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Notes

- The NER model may be downloaded automatically from Hugging Face on first run.
- The classification model is loaded locally from the `models/2kveri` directory.

## Dataset Source

- Kaggle: https://www.kaggle.com/datasets/ulkutuncerkucuktas/turkey-earthquake-relief-tweets-dataset
