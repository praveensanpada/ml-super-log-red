# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv('iris.csv')

# Convert to binary classification problem
df['target'] = df['species'].apply(lambda x: 1 if x == 'Iris-setosa' else 0)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
