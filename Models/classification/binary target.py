import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


data = fetch_california_housing(as_frame=True)
df = data.frame 

median_value = df['MedHouseVal'].median()
df['HighValue']=np.where(df['MedHouseVal']>median_value,1,0)

df = df.drop(columns='MedHouseVal')

print(df['HighValue'].value_counts())

x = df.drop(columns='HighValue')

y = df['HighValue']


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42,stratify=y)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Step 6: Train Logistic Regression
#clf = LogisticRegression(max_iter=1000)
#clf.fit(x_train_scaled, y_train)

# Step 7: Predict and evaluate
#y_pred = clf.predict(x_test_scaled)
#print("LOGISTIC_REGRESSION : ")
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
#print("\nClassification Report:\n", classification_report(y_test, y_pred))


#tree = DecisionTreeClassifier(random_state=42)
#tree.fit(x_train_scaled, y_train)
#y_pred_tree = tree.predict(x_test_scaled)

#print('D-TREE : ')
#print("Accuracy :", accuracy_score(y_test, y_pred_tree))
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
#print("\nClassification Report:\n", classification_report(y_test, y_pred_tree))


rf = RandomForestClassifier(random_state=42)
rf.fit(x_train_scaled, y_train)
y_pred_rf = rf.predict(x_test_scaled)

print("\nRandom-forest :\n ")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))



# Initialize SVM classifier with RBF kernel
#svm_clf = SVC(kernel='rbf', C=1, gamma='scale')  # You can tune these later

# Train
#svm_clf.fit(x_train_scaled, y_train)

# Predict
#y_pred_svm = svm_clf.predict(x_test_scaled)

# Evaluate
#print("\nSVM : \n")
#print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
#print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
#print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(x_train_scaled, y_train)

y_pred = mlp.predict(x_test_scaled)

print("MLP Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

