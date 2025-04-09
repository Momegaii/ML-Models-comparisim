import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



data = fetch_california_housing(as_frame=True)
df = data.frame 




df_clean = df.copy()

df_clean = df_clean[df_clean['AveOccup']< 100]


x = df_clean.drop("MedHouseVal", axis=1)
y = df_clean["MedHouseVal"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("X_train shape : ",x_train_scaled.shape)

print("Example scaled features (first row): ", x_train_scaled[0])


# 1. Train the model
lr_model = LinearRegression()
lr_model.fit(x_train_scaled, y_train)

# 2. Predict
y_pred = lr_model.predict(x_test_scaled)

# 3. Evaluate
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

