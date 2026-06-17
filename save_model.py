import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("creditcard.csv")

X = data.drop(columns=['Class'])
Y = data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, Y_train)

pickle.dump(model, open('fraud_model.pkl', 'wb'))

print("Model Saved Successfully")
