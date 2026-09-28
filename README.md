# Multimodal Wearable Sensor Fusion for Subject-Independent Stress Detection
---

## 📌 About the Project

This project investigates whether combining multiple wearable sensor modalities can improve **subject-independent stress detection using machine learning**.

The study uses the **WESAD (Wearable Stress and Affect Detection)** dataset and evaluates multiple machine learning models using **Leave-One-Subject-Out (LOSO)** validation.

The main research question is:

> **Does multimodal wearable sensor fusion improve stress detection when tested on previously unseen individuals?**

---

## 🎯 Objectives

- Evaluate machine learning models for wearable stress detection.
- Compare individual and multimodal sensor configurations.
- Test model generalization to unseen subjects.
- Analyze the contribution of different sensor modalities.
- Study subject-level variation in model performance.

---

## 📊 Dataset

The project uses the **WESAD (Wearable Stress and Affect Detection)** dataset.

The dataset contains physiological and motion signals including:

- ECG
- EDA
- BVP
- Temperature
- Respiration
- EMG
- 3-axis acceleration

The experiments use data from **15 subjects**.

> **Note:** The raw dataset is **not included** in this repository.

**Dataset sources:**

- [WESAD (University of Siegen)](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/465/wesad%2Awearable%2Astress%2Aand%2Affect%2Adetection)

---

## 🔬 Methodology

WESAD Dataset
      ↓
Data Preprocessing
      ↓
60-Second Windowing
      ↓
ECG / HRV Feature Extraction
      ↓
EDA / Temperature / Acceleration Features
      ↓
Feature Quality Analysis
      ↓
Sensor Configuration
      ↓
Machine Learning
      ↓
LOSO Validation
      ↓
Performance Analysis


### Sensor Configurations

| Configuration | Sensors |
|---|---|
| **C1** | Heart / HRV |
| **C2** | EDA |
| **C3** | Heart / HRV + EDA |
| **C4** | Heart / HRV + EDA + Temperature + Acceleration |

### Machine Learning Models

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

---

## 📈 Key Results

The main analysis was performed using **703 cleaned windows** from **15 subjects**.

### Mean F1-Score (LOSO)

| Configuration | Logistic Regression | SVM | Random Forest | XGBoost |
|---|:---:|:---:|:---:|:---:|
| Heart / HRV | 0.6192 | 0.6520 | 0.6594 | 0.6036 |
| EDA | 0.4385 | 0.5163 | 0.4782 | 0.4934 |
| Heart + EDA | 0.7745 | 0.7412 | 0.7506 | 0.7788 |
| Full Multimodal | 0.7602 | 0.7052 | **0.8167** | 0.7920 |

### Highlights

- The **Full Multimodal** configuration with **Random Forest** achieved the best mean F1-score of **0.8167** under LOSO validation.
- **XGBoost** with the Full Multimodal configuration achieved a mean AUC of approximately **0.9838**.
- Subject-level analysis showed **substantial variation** in performance across individuals.

---

## 📁 Repository Structure


WESAD-STRESS-RESEARCH/
│
├── NOTEBOOKS/
│   ├── 01_explore_wesad.py
│   ├── ...
│   ├── 15_loso_experiment.py
│   ├── 16_confusion_matrix_analysis.py
│   ├── 17_feature_importance_analysis.py
│   ├── 18_statistical_analysis.py
│   └── 19_complete_subject_analysis.py
│
├── RESULTS/
│   ├── complete_subject_analysis/
│   ├── confusion_analysis/
│   ├── feature_importance/
│   ├── statistical_analysis/
│   ├── loso_fold_results.csv
│   └── loso_summary.csv
│
├── .gitignore
└── README.md


The raw WESAD dataset is excluded from the repository using `.gitignore`.

---

## ⚠️ Limitations

- The study uses only **15 subjects**.
- WESAD is a **controlled experimental dataset**.
- No **external dataset validation** has been performed.
- HRV features were extracted from **60-second windows**.
- Subject-level performance varies considerably.
- Motion features may also reflect physical activity or experimental context.

---

## 🔮 Future Work

- External validation on another stress dataset.
- Cross-dataset evaluation.
- Larger and more diverse datasets.
- Improved ECG signal processing.
- Deep learning and temporal models.
- Explainable AI for physiological stress detection.
- Personalized versus subject-independent modeling.

---


## 👨‍💻 Author

**Subham Kumar Singh**
Computer Science Engineering
Nagarjuna College of Engineering and Technology, Bengaluru

GitHub: [Subh10infinity](https://github.com/Subh10infinity)

---

## 📜 Disclaimer

This is an academic research project. The system is **not** a clinically validated stress-detection or medical diagnostic system.
