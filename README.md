# Prediction of Secondary Diagnoses in Clinical Patients Using Multilabel Feedforward Neural Network

**Authors:** Alish Chelackal & Matthias Joos
**Project:** Neural Networks & Deep Learning — Applied Computational Life Science (ZHAW)
**Date:** May 2020 | Data provided in collaboration with the University Hospital of Zurich

---

## Overview

This project develops a **multilabel feedforward neural network (FFNN)** to predict the top five most probable **secondary diagnoses** for hospital patients based on their clinical intake data. The model is trained on real patient records and leverages demographic, diagnostic, and clinical complexity features to support clinical decision-making.

Secondary diagnosis prediction is a clinically meaningful challenge — accurate anticipation of comorbidities can improve treatment planning, resource allocation, and patient outcomes in hospital settings.

---

## Clinical Context

| Feature | Description |
|---|---|
| `Hauptdiagnose` | Primary (main) diagnosis — ICD code |
| `Nebendiagnose` | Secondary diagnosis — prediction target |
| `MDC` | Major Diagnostic Category |
| `PCCL` | Patient Clinical Complexity Level (0–4) |
| `DRG` | Diagnosis Related Group |
| `Partition` | Case type: Surgical (O), Medical (M), Other (0) |
| `Alter` | Patient age |
| `Geschlecht` | Patient sex |

---

## Model Architecture

A sequential Keras FFNN with the following design choices:

- **Input layer:** Mixed feature types handled via TensorFlow Feature Columns
  - Numerical features (Age, MDC, PCCL) → Min-max normalised
  - Categorical features (Sex, Partition) → One-hot encoded
  - High-cardinality features (Main Diagnosis, DRG) → Embedding encoded (dim=8)
- **Hidden layers:** Dense(15, ReLU) → Dropout(0.1) → Dense(32, ReLU) → BatchNorm → Dropout(0.1)
- **Output layer:** Dense(n_classes, Sigmoid) — one probability per unique secondary diagnosis
- **Loss:** Binary cross-entropy (multilabel classification)
- **Optimizer:** Adam (lr=0.01)

---

## Key Features

- **Multilabel classification** — predicts probability scores for all known secondary diagnoses simultaneously
- **Top-5 prediction** — outputs the five most probable secondary diagnoses per patient
- **User-defined input** — interactive CLI function allows custom patient profiles to be entered and predicted in real time
- **Training visualisation** — accuracy and loss curves plotted across epochs for both train and validation sets

---

## Results

The model was trained for 10 epochs with an 80/20 train-test split (validation carved from training set). Training and validation accuracy/loss curves are generated automatically for performance assessment.

---

## Requirements
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

Python 3.7+ recommended. TensorFlow 2.x.

---

## Usage
```bash
# Clone the repository
git clone https://github.com/your-username/secondary-diagnosis-prediction.git
cd secondary-diagnosis-prediction

# Run the model
python secondary_diagnosis_ffnn.py
```

To test with your own patient input, uncomment the last line in the script:
```python
print(user_input())
```
You will be prompted to enter sex, partition, main diagnosis, DRG, age, MDC, and PCCL values interactively.

---

## Project Structure
```
├── secondary_diagnosis_ffnn.py   # Main model script
├── diagnoses.csv                 # Patient dataset (not included — confidential clinical data)
└── README.md
```

> **Note:** The original patient dataset is not included in this repository as it contains confidential clinical records from the University Hospital of Zurich.

---

## Relevance

This project demonstrates applied deep learning in a real clinical setting, including:
- Handling heterogeneous, high-dimensional medical data
- Multilabel classification for clinical decision support
- Feature engineering for mixed data types (categorical, numerical, high-cardinality)
- Collaboration with clinical data from a Swiss university hospital

---

## Authors

**Alish Chelackal** — [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)
**Matthias Joos**

MSc Applied Computational Life Science (Data Science), ZHAW, Switzerland
