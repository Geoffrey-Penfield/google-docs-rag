import pickle
import os

if os.path.exists("./docs.pkl"):
    with open('./docs.pkl', 'rb') as file:
        docs = pickle.load(file)

print(docs[0])