# Evaluating LLMs for Sequential Skip Prediction in Music Streaming

**MSc Dissertation — Data Science & AI, University of Liverpool (2023–24)**
**Grade: 83 | Overall Award: Pass with Distinction**

## Overview

This project investigates whether large language models (LLMs) can predict sequential skip behaviour in music streaming sessions — that is, given a user's listening history within a session, can an LLM correctly predict whether the next track will be skipped?

The work spans the full research pipeline: data preprocessing at scale, systematic prompt engineering across ten experimental conditions, multi-model inference (Gemini 1.5 Flash and Meta's Llama 3.0/3.1), and evaluation against ground-truth labels using two custom accuracy metrics.

## Research Question

> Can LLMs leverage information about a user's in-session listening history and/or audio content features to predict sequential skip behaviour more accurately than a majority-class baseline?

## Dataset

Experiments were conducted on the **Spotify Million Session Dataset (MSSD)**, an industry-scale dataset comprising:
- Listening session logs (~130M rows) with skip labels, session metadata, and behavioural signals
- Track feature data (~3.7M tracks) with audio content features (acousticness, energy, danceability, etc.)

**The full MSSD is not included in this repository** due to Spotify licensing restrictions. Sample files are provided in `data/sample/` for code testing purposes. The full dataset is available from [Spotify AI Research](https://research.atspotify.com/2020/09/the-spotify-million-playlist-dataset-challenge/).

## Models

| Model | Interface | Deployment |
|---|---|---|
| Gemini 1.5 Flash | Google Generative AI API | Google Cloud Platform |
| Meta Llama 3.0 8B Instruct | HuggingFace Transformers | GCP / HPC cluster |
| Meta Llama 3.1 8B Instruct | HuggingFace Transformers | GCP / HPC cluster |
| Majority-class baseline (dummy) | — | Local |

## Experimental Design

Ten experimental conditions were defined by varying two dimensions:

**Information type provided in the prompt:**
- Experiment 1: Track metadata only (name, artist, album)
- Experiment 2: Track metadata + audio content features
- Experiment 3: Track metadata + session-level behavioural signals
- Experiment 4: Track metadata + audio features + behavioural signals
- Experiment 5: Track metadata + audio features + behavioural signals + explicit skip history labels

**Prompting strategy:**
- Zero-shot: no example of the expected response format
- One-shot: one labelled example prepended to the prompt

Each condition was run across **three random seeds** and results aggregated for robustness. Approximately **35,000 prompts** were issued in total.

## Evaluation Metrics

- **Session-level accuracy**: proportion of sessions in which the model correctly predicts the skip/not-skip outcome for the final track
- **Positional accuracy**: accuracy broken down by position within the session, to detect positional biases

## Key Findings

- Gemini 1.5 Flash consistently outperformed both Llama models and the majority-class dummy classifier across most experimental conditions
- One-shot prompting eliminated catastrophic prediction failures (runs where the model returned only one label regardless of input) but introduced **semantic contamination** in certain conditions — the model appeared to pattern-match from the one-shot example rather than reason from the input history
- Llama models produced unstable outputs under zero-shot conditions, often failing to conform to the required binary response format despite constrained generation settings
- Audio content features provided limited additional signal beyond track metadata for skip prediction in this formulation

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── runner.py                           # Main inference script (all prompt templates, both models)
├── gemini_main.py                      # Standalone Gemini inference script
├── gemini_prompting.py                 # Minimal Gemini prompting test
├── llama_main_for_colab.py             # Llama inference script (Colab deployment)
├── Sample listening log for testing.csv
├── Sample track features for testing.csv
├── session_accuracy_results_prelims.csv
├── positional_accuracy_results_prelims.csv
├── logs/                               # Experiment logs from Gemini runs
├── Project Results/
│   └── ProjectResults2/
│       ├── Tables/                     # Final aggregated results tables
│       ├── gemini/                     # Per-seed Gemini results
│       ├── llama3_0/                   # Per-seed Llama 3.0 results
│       ├── llama3_1/                   # Per-seed Llama 3.1 results
│       └── dummy/                      # Majority-class baseline results
└── Source Code/
    ├── mssd_test_data_analysis.ipynb   # Exploratory data analysis on the MSSD
    ├── gemini_main_on_cloud.ipynb      # Gemini inference notebook (GCP/Colab)
    ├── Llama_for_colab_main.ipynb      # Llama inference notebook (Colab)
    ├── llama_main_for_HPC.py           # Llama inference script (HPC cluster)
    └── Evaluation.ipynb                # Results aggregation and analysis
```

## How to Run

### Requirements

```bash
pip install -r requirements.txt
```

### API Keys

Create a `.env` file in the project root:

```
YOUR_API_KEY=your_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here
```

Gemini API access: [Google AI Studio](https://aistudio.google.com/)
Llama model access: requires HuggingFace account with Meta Llama access granted

### Data

The GCP-based notebooks (`gemini_main_on_cloud.ipynb`, `Llama_for_colab_main.ipynb`) read data from a GCS bucket. To run locally, update the file paths in `runner.py` (`load_data()`) to point to a local copy of the MSSD. Sample files in `data/sample/` can be used for testing the pipeline with a small subset.

### Running an Experiment

```bash
python runner.py --seed 42 --model gemini --experiment 1 --shot_type zero
```

Arguments:
- `--seed`: random seed for reproducibility
- `--model`: `gemini` or `llama`
- `--experiment`: experiment number (1–5)
- `--shot_type`: `zero` or `one`

### Notebooks

The notebooks in `Source Code/` were developed on Google Colab with GCP storage. They contain inline outputs from the original experimental runs and serve primarily as documentation of the research process. The `Evaluation.ipynb` notebook aggregates results and produces the figures and tables reported in the dissertation.

## Technical Stack

- Python 3.10
- `google-generativeai` — Gemini 1.5 Flash API
- `transformers`, `torch` — Llama 3.0/3.1 inference via HuggingFace
- `pandas`, `numpy` — data handling
- `matplotlib`, `seaborn` — visualisation
- `python-dotenv` — API key management
- Google Cloud Platform (GCS, Colab, Vertex AI)
