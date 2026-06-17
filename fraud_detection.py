import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("creditcard.csv")

print(data.head())

# Separate Features and Target
X = data.drop(columns=['Class'])
Y = data['Class']

# Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)

# Training Accuracy
X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(
    X_train_prediction,
    Y_train
)

print("Training Accuracy:", training_accuracy)

# Testing Accuracy
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(
    X_test_prediction,
    Y_test
)

print("Testing Accuracy:", test_accuracy)
