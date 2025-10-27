<div align="center">
  
# 🧬 Predicting DNA Binding Proteins with Machine Learning & Deep Learning

</div>

This project investigates the classification of **DNA-binding proteins (DBPs)** using classical machine learning and deep learning models. DBPs are crucial to processes like transcription and DNA repair, and accurately identifying them has implications in **drug design, bioinformatics, and molecular biology**.


## 🔍 Overview

We compare several supervised learning models—Logistic Regression, k-Nearest Neighbours (kNN), XGBoost—and a deep learning architecture combining 1D Convolutional and Bidirectional LSTM layers. Models are evaluated using Accuracy, Sensitivity, Specificity, and Matthews Correlation Coefficient (MCC).

## 📦 Dataset

- **Train set**: 14,926 protein sequences  
- **Test set**: 57,111 protein sequences  
- Each sequence contains amino acid characters (20 unique types)  
- Labels: `1` for DNA-binding, `0` for non-binding  
- Balanced training set (50.3% positive), but imbalanced test set (28.7% positive)

## ⚙️ Methods

### Feature Extraction
- **k-mer tokenization**: Sequences are decomposed into overlapping substrings of length `k`
- **Vectorization**: Applied `CountVectorizer` to generate frequency-based features
- **Variance Thresholding**: Low-variance features were removed to reduce noise

### Model Training
- **ML models**: Hyperparameters tuned using `GridSearchCV`, optimized for MCC  
- **DL model**: One-hot encoding + CNN + Bi-LSTM + Dense layers  
  - Sequence length standardized to 1500
  - Dropout and ReLU activation used to prevent overfitting

## 📊 Result Overview

| Model              | Accuracy | Sensitivity | Specificity | MCC   |
|-------------------|----------|-------------|-------------|--------|
| k-Nearest Neighbour | 0.71     | 0.26        | **0.90**    | 0.20  |
| Logistic Regression | 0.59     | 0.66        | 0.57        | 0.21  |
| XGBoost            | 0.66     | **0.71**    | 0.53        | **0.23**  |
| Deep Learning      | 0.60     | 0.51        | 0.62        | 0.10  |

- **XGBoost** had the highest MCC (0.23) and best balance overall
- **kNN** achieved strong specificity but weak sensitivity
- **DL model** underperformed due to noise and overfitting risk


## 📌 Key Learnings
- Sequence-based classification is viable using simple k-mer representations
- XGBoost performs robustly even on imbalanced sequence datasets
- Deep learning requires more tuning and data volume to generalize well in this context

## 👥 Contributors
- Ryan Phua Wei Jie  
- Noel Teo Yu Hong  
- Lucius Khor Zhe Yi  
- Del Mundo Nino Hubert Atienza

---


