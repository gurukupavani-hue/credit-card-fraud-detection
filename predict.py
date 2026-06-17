import pickle

model = pickle.load(open('fraud_model.pkl', 'rb'))

print("Model Loaded Successfully")
