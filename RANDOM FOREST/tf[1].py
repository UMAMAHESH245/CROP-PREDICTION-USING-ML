import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Read the CSV file
crop = pd.read_csv('Crop_recommendation.csv')
print(crop.head())
print(crop.shape)
print(crop.isnull().sum())
print(crop.duplicated().sum())
print(crop.info())
print(crop.describe())

# Group data by 'label' and plot the mean values
grouped = crop.groupby("label")
grouped.mean()["N"].plot(kind="barh")
plt.show()
grouped.mean()["P"].plot(kind="barh")
plt.show()
grouped.mean()["K"].plot(kind="barh")
plt.show()
grouped.mean()["temperature"].plot(kind="barh")
plt.show()
grouped.mean()["rainfall"].plot(kind="barh")
plt.show()
grouped.mean()["humidity"].plot(kind="barh")
plt.show()
grouped.mean()["ph"].plot(kind="barh")
plt.show()

# Create a dictionary to map crop labels to numeric values
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9,
    'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Map the 'label' column to numeric values and drop the original column
crop['label_num'] = crop['label'].map(crop_dict)
crop.drop('label', axis=1, inplace=True)
print(crop.head())

# Split the dataset into features (X) and labels (y)
X = crop.iloc[:, :-1]
y = crop.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train different machine learning models and evaluate their accuracy
models = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name}:\nAccuracy: {acc:.4f}')

# Selecting the Random Forest model for predictions
rfc = RandomForestClassifier()
rfc.fit(X_train_scaled, y_train)
y_pred = rfc.predict(X_test_scaled)
print(accuracy_score(y_test, y_pred))


# Define a function to make crop predictions
def predict_crop(N, P, K, temperature, humidity, pH, rainfall):
    input_values = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    prediction = rfc.predict(input_values)
    return prediction[0]


# Sample input values for prediction
N = 114
P = 21
K = 55
tem = 25.44
humidity = 87.94
ph = 6.47
rainfall = 257.52

pred = predict_crop(N, P, K, tem, humidity, ph, rainfall)

# Map the numeric prediction back to crop labels
predicted_crop = {v: k for k, v in crop_dict.items()}

if pred in predicted_crop:
    print(f"{predicted_crop[pred]} is the best crop to be cultivated right there")
else:
    print("Sorry, we could not determine the best crop to be cultivated with the provided data.")

# Save the trained Random Forest model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(rfc, model_file)

# X_train
