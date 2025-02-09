import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from nltk.corpus import stopwords
import re
import nltk

# Descarcă lista de stop words dacă nu este deja descărcată
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Funcția de preprocesare a textului
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Eliminăm caracterele non-alfabetice și transformăm în litere mici
    tokens = text.split()  # Împărțim textul în cuvinte
    tokens = [word for word in tokens if word not in stop_words]  # Eliminăm stop word-urile
    return ' '.join(tokens)

# Incarcarea datelor
train = pd.read_csv('Date/train.csv')
test = pd.read_csv('Date/test.csv')
val = pd.read_csv('Date/val.csv')

# Curatarea textului
train["text"] = train["text"].apply(preprocess_text)
val["text"] = val["text"].apply(preprocess_text)
test["text"] = test["text"].apply(preprocess_text)

# Adaugarea unui zgomot aleator scorurilor tinta
noise_level = 0.23
train["score"] += np.random.normal(0, noise_level, size=len(train))

# Vectorizarea textului
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), min_df=2)
x_train = vectorizer.fit_transform(train["text"])
x_val = vectorizer.transform(val["text"])
x_test = vectorizer.transform(test["text"])

# Scorurile tinta
y_train = train["score"]
y_val = val["score"]

# Antrenarea modelului
model = Ridge(alpha=1)
model.fit(x_train, y_train)

# Predictii
val_predictions = model.predict(x_val)
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

# Predictii pe setul de test
test_predictions = model.predict(x_test)

# Salvarea rezultatelor
submission = pd.DataFrame({
    "id": test["id"],
    "score": test_predictions
})
submission.to_csv("DateOUT/IonutPantiru.csv", index=False)

# Get feature importance
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_.flatten()

# Sort by absolute value
sorted_indices = np.argsort(np.abs(coefficients))[::-1]
top_features = feature_names[sorted_indices[:20]]
top_weights = coefficients[sorted_indices[:20]]

# Bar chart
plt.figure(figsize=(10, 6))
plt.barh(top_features, top_weights, color='skyblue')
plt.xlabel('Coefficient Weight')
plt.ylabel('Feature')
plt.title('Top 20 Features by Weight')
plt.gca().invert_yaxis()
plt.show()

