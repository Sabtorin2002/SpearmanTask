# Practical Machine Learning: Visual Sentence Complexity

This project explores machine learning techniques to predict scores based on textual data. It incorporates advanced preprocessing, feature extraction, and model optimization for accurate predictions.

## Table of Contents
- [Introduction](#introduction)
- [Models Used](#models-used)
- [Methodology](#methodology)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
The goal of this project is to predict scores from textual data using a combination of natural language processing and regression models. Key features include:
- Extensive preprocessing (e.g., stopword removal, tokenization, text cleaning).
- Feature extraction using TF-IDF and Word2Vec.
- Evaluation of various regression models like Ridge, XGBoost, LightGBM, Random Forest, K-NN, and SVM.

## Models Used
### Ridge Regression
- Applied with TF-IDF for feature extraction.
- Handles multicollinearity and reduces overfitting with regularization.

### XGBoost and LightGBM
- Gradient boosting algorithms optimized for computational efficiency.
- XGBoost was selected for its scalability and performance with large datasets.
- LightGBM emerged as the best-performing model due to its gradient-based optimization and reduced training time.

### Word2Vec
- Converts text into semantic vectors using CBOW and Skip-Gram architectures.

### Other Models
- Random Forest, K-NN, and SVM were also explored, but they underperformed compared to LightGBM.

## Methodology
1. **Text Preprocessing**:
   - Removing non-alphabetic characters.
   - Lowercasing and tokenization.
   - Removing stopwords using NLTK.
   
2. **Feature Extraction**:
   - TF-IDF: Extracts text features based on term frequency and document frequency.
   - Word2Vec: Embeds words into continuous vector space for semantic representation.
   
3. **Hyperparameter Optimization**:
   - GridSearchCV was used for hyperparameter tuning.
   - Key parameters like learning rate, max depth, and n_estimators were optimized for XGBoost and LightGBM.

4. **Evaluation**:
   - Models were evaluated using metrics such as Spearman's Rank Correlation, Mean Absolute Error (MAE), and Mean Squared Error (MSE).

## Results
- Ridge with TF-IDF provided initial results, but performance improved significantly with Word2Vec and XGBoost/LightGBM.
- LightGBM achieved the best performance with optimal hyperparameters:
  - `learning_rate = 0.01`
  - `max_depth = 7`
  - `n_estimators = 400`
  - `subsample = 0.8`

- Evaluation Metrics:
  - Spearman Coefficient: Improved with Word2Vec and LightGBM.
  - Learning curves showed reduced error with larger datasets.

## Requirements
Install the required Python libraries:
```bash
pip install pandas numpy nltk scikit-learn xgboost lightgbm gensim

## Conclusion
This project demonstrated the power of advanced text preprocessing and gradient boosting models in solving textual regression tasks. 
LightGBM combined with Word2Vec achieved the best results, highlighting the importance of tailored preprocessing and model selection for textual data.


## References 
Ridge Regression - GeeksforGeeks
TF-IDF - GeeksforGeeks
XGBoost - GeeksforGeeks
LightGBM - GeeksforGeeks
Word2Vec - GeeksforGeeks
