"""
Sentiment Analysis using Random Forest Classifier
This script performs sentiment analysis on Amazon food reviews using TF-IDF vectorization
and Random Forest classifier with hyperparameter tuning.
"""

# Install required packages (uncomment if needed)
# pip install pandas scikit-learn wordcloud matplotlib nltk

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download stopwords and wordnet (only needed once)
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the dataset
print("\nLoading dataset...")
file_path = 'foods_testing.csv'
df = pd.read_csv(file_path, sep=',', quotechar='"', encoding='ISO-8859-1')

# Check the first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head())

# Text Preprocessing: Clean, remove stop words, and lemmatize
def preprocess_text(text):
    """
    Preprocess text by removing punctuation, converting to lowercase,
    removing stopwords, and lemmatizing.
    
    Args:
        text: Input text string
        
    Returns:
        Preprocessed text string
    """
    # Remove punctuation and non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()  # Convert to lowercase
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back into a single string
    return " ".join(words)

# Apply text preprocessing
print("\nPreprocessing text...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Vectorize the text using TF-IDF
print("\nVectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=2000)  # Increased feature count
X = vectorizer.fit_transform(df['cleaned_text'])

# Assuming binary sentiment labels (you should use real sentiment labels)
# NOTE: This is using random labels for illustration purposes only
# In a real scenario, you would use actual sentiment labels from your dataset
print("\nCreating sentiment labels...")
df['sentiment'] = np.random.randint(0, 2, size=len(df))  # Random sentiment for illustration
y = df['sentiment']

# Split the data
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for Random Forest
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [10, 20, 30],        # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split
    'min_samples_leaf': [1, 2, 4],    # Minimum samples required at a leaf node
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV to find best hyperparameters
print("Running GridSearchCV (this may take a while)...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters from GridSearchCV
best_rf = grid_search.best_estimator_
print(f"\nBest hyperparameters: {grid_search.best_params_}")

# Train the model with best hyperparameters
print("\nTraining model with best hyperparameters...")
best_rf.fit(X_train, y_train)

# Predict and evaluate
print("\nEvaluating model...")
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy after tuning: {accuracy:.3f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
print("\nGenerating confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# Feature Importance Visualization
print("\nGenerating feature importance plot...")
feature_importance = best_rf.feature_importances_
sorted_idx = feature_importance.argsort()

# Plot top 20 important features
plt.figure(figsize=(10, 8))
plt.barh(range(20), feature_importance[sorted_idx][-20:], align='center')
plt.yticks(range(20), [vectorizer.get_feature_names_out()[i] for i in sorted_idx[-20:]])
plt.xlabel('Feature Importance')
plt.title('Top 20 Important Features for Sentiment Classification')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("Feature importance plot saved as 'feature_importance.png'")
plt.show()

print("\nAnalysis complete!")
