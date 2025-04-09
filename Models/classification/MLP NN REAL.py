
import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier


# Load data
data = fetch_california_housing(as_frame=True)
df = data.frame


# Create binary target
median_value = df['MedHouseVal'].median()
df['HighValue'] = np.where(df['MedHouseVal'] > median_value, 1, 0)
df.drop(columns='MedHouseVal', inplace=True)

X = df.drop(columns='HighValue')
y = df['HighValue']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




mlp_tanh = MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', max_iter=500, random_state=42)
mlp_tanh.fit(X_train_scaled, y_train)

y_pred = mlp_tanh.predict(X_test_scaled)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


