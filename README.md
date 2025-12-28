
# Water Potability Classification

This repository contains a machine learning project that predicts whether drinking water is **potable** (safe to drink) based on its physicochemical properties. The model is trained on the Kaggle Water Potability dataset with 3,276 samples and 9 numeric features.

## Project Overview

- Goal: Build a decision‑support tool to screen water samples before laboratory testing, not to replace certified lab analysis.  
- Models used: Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM), all sharing the same preprocessing pipeline.  
- Best performance: SVM and Random Forest achieve around 66–67% test accuracy with ROC‑AUC values close to 0.65, indicating better‑than‑random but moderate performance.

## Dataset

- Source: Kaggle “Water Potability” dataset (CSV file: `water_potability.csv`).  
- Samples: 3,276 observations.  
- Features (9):  
  - pH  
  - Hardness  
  - Solids  
  - Chloramines  
  - Sulfate  
  - Conductivity  
  - Organic Carbon  
  - Trihalomethanes  
  - Turbidity  
- Target variable: `Potability` (0 = not safe, 1 = safe), with more non‑potable than potable samples, leading to class imbalance.

## Methods

### 1. Exploratory Data Analysis (EDA)

- Plotted histograms of each feature to inspect distributions.  
- Created a correlation heatmap to understand relationships between variables and the target.

### 2. Preprocessing

- Handled missing values (notably in pH, Sulfate, Trihalomethanes) using median imputation for all numeric features.  
- Applied `StandardScaler` to normalize feature ranges.  
- Performed an 80–20 stratified train–test split to maintain the original class distribution.

### 3. Modeling

- Built pipelines that combine preprocessing and each classifier:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest (around 200 trees)  
  - SVM with probability estimates enabled  
- Evaluated models using:  
  - Accuracy  
  - Precision, Recall, F1‑score  
  - Confusion Matrix  
  - ROC‑AUC and ROC curves

## Results

| Model               | Test Accuracy | ROC‑AUC |
|---------------------|---------------|---------|
| Logistic Regression | ≈ 0.61        | ≈ 0.55  |
| Decision Tree       | ≈ 0.60        | ≈ 0.57  |
| Random Forest       | ≈ 0.66        | ≈ 0.65  |
| SVM                 | ≈ 0.67        | ≈ 0.65  |

- SVM achieved the highest accuracy, with Random Forest very close and offering feature importance for interpretability.  
- Confusion matrices show that models tend to correctly classify non‑potable water more often than potable water, reflecting the underlying class imbalance.  
- Overall, performance is moderate: useful for prioritising which samples should be tested in a lab, but not reliable enough for standalone certification.

## Suggested Repository Structure

You can adapt this to your actual layout:

- `notebooks/Water_Quality_Classification.ipynb` – main notebook with EDA, preprocessing, modeling, and evaluation.  
- `data/water_potability.csv` – raw dataset (download from Kaggle).  
- `data/clean_water_potability.csv` – cleaned dataset with imputed values.  
- `models/water_potability_model.pkl` – saved best model for deployment.  
- `reports/water-report.pdf` – detailed written report.  
- `slides/Water-Potability-Classification.pptx` – presentation slides.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Place `water_potability.csv` in the `data/` directory.

3. Run the notebook or training script to train and evaluate the models:

```bash
python train_models.py
```

4. Load the saved model and make predictions on new samples:

```python
import joblib
import pandas as pd

model = joblib.load("models/water_potability_model.pkl")

# Example: one sample with the 9 features
sample = pd.DataFrame([{
    "ph": 7.0,
    "Hardness": 200.0,
    "Solids": 20000.0,
    "Chloramines": 7.0,
    "Sulfate": 350.0,
    "Conductivity": 450.0,
    "Organic_carbon": 15.0,
    "Trihalomethanes": 70.0,
    "Turbidity": 3.5
}])

prediction = model.predict(sample)
print("Potable" if prediction[0] == 1 else "Not potable")
```

## Limitations and Future Work

- Class imbalance (more non‑potable samples) reduces recall for potable water; this could be improved using class‑weighting or resampling techniques such as SMOTE.  
- Dataset size and feature scope are limited (no temporal, geographic, or microbiological data), which caps maximum performance.  
- Future improvements may include:  
  - Hyperparameter tuning (grid search for SVM and Random Forest).  
  - Domain‑informed feature engineering (e.g., pH deviation from neutrality, hardness bins).  
  - Ensemble approaches combining SVM and Random Forest.  
