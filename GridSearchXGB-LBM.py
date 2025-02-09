import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from gensim.models import Word2Vec
import re
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

# Descarcă lista de stop words dacă nu este deja descărcată
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocesarea textului
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Eliminăm caracterele non-alfabetice și transformăm în litere mici
    tokens = text.split()  # Împărțim textul în cuvinte
    tokens = [word for word in tokens if word not in stop_words]  # Eliminăm stop word-urile
    return tokens

# Încărcarea datelor
train = pd.read_csv('Date/train.csv')
test = pd.read_csv('Date/test.csv')
val = pd.read_csv('Date/val.csv')

# Aplicarea preprocesării
train["tokens"] = train["text"].apply(preprocess_text)
val["tokens"] = val["text"].apply(preprocess_text)
test["tokens"] = test["text"].apply(preprocess_text)

# Antrenarea unui model Word2Vec pe textul tokenizat
all_tokens = (train["tokens"].tolist() + val["tokens"].tolist() + test["tokens"].tolist())
word2vec_model = Word2Vec(sentences=all_tokens, vector_size=100, window=4, min_count=2, workers=4, sg=0, epochs=25)

# Funcție pentru a calcula vectorul mediu pentru fiecare propoziție
def compute_sentence_vector(tokens, model, vector_size):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

# Calcularea vectorilor pentru seturile de date
vector_size = word2vec_model.vector_size
x_train = np.array([compute_sentence_vector(tokens, word2vec_model, vector_size) for tokens in train["tokens"]])
x_val = np.array([compute_sentence_vector(tokens, word2vec_model, vector_size) for tokens in val["tokens"]])
x_test = np.array([compute_sentence_vector(tokens, word2vec_model, vector_size) for tokens in test["tokens"]])

# Scorurile țintă
y_train = train["score"]
y_val = val["score"]

# Definirea unui estimatori personalizați pentru a integra XGBoost și LightGBM
class MultiModelRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_type="xgboost", **params):
        self.model_type = model_type
        self.params = params
        self.model = None

    def set_params(self, **params):
        """Updates model-specific parameters."""
        if 'model_type' in params:
            self.model_type = params.pop('model_type')
        self.params.update(params)
        return self

    def fit(self, X, y):
        if self.model_type == "xgboost":
            self.model = XGBRegressor(**self.params, random_state=42)
        elif self.model_type == "lightgbm":
            self.model = LGBMRegressor(**self.params, random_state=42)
        else:
            raise ValueError("Unsupported model_type. Use 'xgboost' or 'lightgbm'.")
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Setul de hiperparametri pentru GridSearchCV
param_grid = [
    {
        # 'model__n_estimators': [400],
        # 'model__learning_rate': [0.04],
        # 'model__max_depth': [10],
        # 'model__subsample': [0.8],
        # 'model__colsample_bytree': [1.0],
        'model__model_type': ['xgboost'],
        'model__n_estimators': [100, 200, 300, 400],
        'model__learning_rate': [0.01, 0.05, 0.04, 0.1, 0.08, 0.13],
        'model__max_depth': [5, 7, 10],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0]
    },
    {
        # 'model__n_estimators': [100],
        # 'model__learning_rate': [0.1],
        # 'model__max_depth': [7],
        # 'model__num_leaves': [31],
        # 'model__min_child_samples': [10],
        # 'model__subsample': [0.8],
        'model__model_type': ['lightgbm'],
        'model__n_estimators': [100, 200, 300, 400],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.04, 0.08, 0.13],
        'model__max_depth': [5, 7, 10],
        'model__num_leaves': [31, 50, 70],
        'model__min_child_samples': [10, 20],
        'model__subsample': [0.8, 1.0],
    }
]

# Definirea pipeline-ului
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalizarea datelor
    ('model', MultiModelRegressor())  # Alegerea modelului
])

# Configurarea GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Executarea Grid Search
grid_search.fit(x_train, y_train)

# Cei mai buni hiperparametri
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Modelul optim
best_model = grid_search.best_estimator_

# Predicții pe setul de validare
val_predictions = best_model.predict(x_val)
spearman_corr, _ = spearmanr(y_val, val_predictions)
print(f"Spearman's Rank Correlation Coefficient on validation set: {spearman_corr:.4f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import kendalltau

mae = mean_absolute_error(y_val, val_predictions)
mse = mean_squared_error(y_val, val_predictions)
kendall_corr, _ = kendalltau(y_val, val_predictions)

print(f"Mean Absolute Error (MAE) on validation set: {mae:.4f}")
print(f"Mean Squared Error (MSE) on validation set: {mse:.4f}")
print(f"Kendall's Tau Correlation Coefficient on validation set: {kendall_corr:.4f}")

# Predicții pe setul de testare
test_predictions = best_model.predict(x_test)

# Salvarea rezultatelor
submission = pd.DataFrame({
    "id": test["id"],
    "score": test_predictions
})
submission.to_csv("DateOUT/OlimpiuMorutan.csv", index=False)

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# Generate learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    estimator=grid_search.best_estimator_,
    X=x_train,
    y=y_train,
    cv=3,
    scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

# Calculate mean and standard deviation for training and validation scores
train_scores_mean = -np.mean(train_scores, axis=1)  # Negative MSE -> Positive MSE
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = -np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Plotting the learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label="Training Error", color="blue")
plt.fill_between(train_sizes,
                 train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std,
                 alpha=0.2, color="blue")
plt.plot(train_sizes, val_scores_mean, label="Validation Error", color="green")
plt.fill_between(train_sizes,
                 val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std,
                 alpha=0.2, color="green")

# Chart labels and legend
plt.title("Learning Curve for Best Model")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Squared Error")
plt.legend(loc="upper right")
plt.grid()
plt.show()

