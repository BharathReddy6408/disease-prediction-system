import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load training and testing data
df = pd.read_csv("Training.csv")
df_test = pd.read_csv("Testing.csv")

# Encode target variable
le = preprocessing.LabelEncoder()
df["prognosis"] = le.fit_transform(df["prognosis"])
df_test["prognosis"] = le.transform(df_test["prognosis"])

X_train = df.drop(columns=["prognosis"])
y_train = df["prognosis"]
X_test = df_test.drop(columns=["prognosis"])
y_test = df_test["prognosis"]

# Get all symptoms and diseases
symptoms = X_train.columns.tolist()
diseases = le.classes_

# Helper function to create input vector from selected symptoms
def get_input_vector(psymptoms):
    input_vector = [0] * len(symptoms)
    for symptom in psymptoms:
        if symptom in symptoms:
            input_vector[symptoms.index(symptom)] = 1
    return input_vector

# Initialize classifiers
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "NaiveBayes": GaussianNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

# Train all models
for model in models.values():
    model.fit(X_train, y_train)

# GUI Setup
root = Tk()
root.title("A MACHINE LEARNING-BASED SYSTEM FOR SYMPTOM-DRIVEN DISEASE PREDICTION")
root.configure(background='black')
root.geometry("700x600")

# Heading
Label(root, text="A MACHINE LEARNING-BASED SYSTEM FOR SYMPTOM-DRIVEN DISEASE PREDICTION", font=("Helvetica", 20, "bold"), bg="lightblue").pack(pady=10)

# Patient Info Frame
patient_frame = Frame(root, bg="black")
patient_frame.pack(pady=10)

Label(patient_frame, text="Name:", fg="white", bg="black", font=("Helvetica", 12, "bold")).grid(row=0, column=0, padx=10)
NameEn = Entry(patient_frame)
NameEn.grid(row=0, column=1)

# Symptoms Frame
symptom_frame = Frame(root, bg="black")
symptom_frame.pack(pady=10)

symptom_vars = []
for i in range(5):
    Label(symptom_frame, text=f"Symptom {i+1}:", fg="white", bg="black", font=("Helvetica", 12, "bold")).grid(row=i, column=0, padx=10, pady=5)
    var = StringVar()
    option = OptionMenu(symptom_frame, var, *symptoms)
    option.config(width=30)
    option.grid(row=i, column=1, padx=10)
    symptom_vars.append(var)

# Result Labels
result_labels = {}
for i, model_name in enumerate(models.keys()):
    Label(root, text=f"{model_name} Prediction:", fg="white", bg="black", font=("Helvetica", 12, "bold")).pack()
    result_labels[model_name] = Label(root, text="", bg="white", fg="black", font=("Helvetica", 12))
    result_labels[model_name].pack(pady=2)

# Predict function
def predict_disease():
    psymptoms = [var.get() for var in symptom_vars if var.get()]
    if not psymptoms:
        for lbl in result_labels.values():
            lbl.config(text="Please select at least one symptom")
        return

    input_vector = [get_input_vector(psymptoms)]
    for model_name, model in models.items():
        pred = model.predict(input_vector)[0]
        disease = diseases[pred]
        result_labels[model_name].config(text=disease)

# Predict Button
Button(root, text="Predict Disease", command=predict_disease, bg="green", fg="white", font=("Helvetica", 14, "bold")).pack(pady=20)

root.mainloop()
