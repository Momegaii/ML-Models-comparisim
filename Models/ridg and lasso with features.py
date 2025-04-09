import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
import pandas as pd


data = fetch_california_housing(as_frame=True)
df = data.frame 

print(df.head())

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

print(" ")


# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_train_scaled, y_train)
ridge_preds = ridge_model.predict(x_test_scaled)

ridge_mae = mean_absolute_error(y_test, ridge_preds)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
ridge_r2 = r2_score(y_test, ridge_preds)

print("Ridge Regression(TUNED):")
print(f"MAE: {ridge_mae:.4f}, RMSE: {ridge_rmse:.4f}, R²: {ridge_r2:.4f}")

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train_scaled, y_train)
lasso_preds = lasso_model.predict(x_test_scaled)

lasso_mae = mean_absolute_error(y_test, lasso_preds)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_preds))
lasso_r2 = r2_score(y_test, lasso_preds)

print("\nLasso Regression (TUNED ):")
print(f"MAE: {lasso_mae:.4f}, RMSE: {lasso_rmse:.4f}, R²: {lasso_r2:.4f}")

x_fe = x.copy()



x_fe['Distance_to_LA'] = np.sqrt((x_fe['Latitude'] - 34.05)**2 + (x_fe['Longitude'] + 118.25)**2)
# 1. Average rooms per household
x_fe['RoomsPerHousehold'] = x_fe['AveRooms'] / x_fe['AveOccup']

# 2. Average bedrooms per room
x_fe['BedrmsPerRoom'] = x_fe['AveBedrms'] / x_fe['AveRooms']

# 3. Population per household
x_fe['PopulationPerHousehold'] = x_fe['Population'] / x_fe['AveOccup']

# 4. Income per household size
x_fe['IncomePerHousehold'] = x_fe['MedInc'] / x_fe['AveOccup']

x_fe['MedInc_HouseAge'] = x_fe['MedInc'] * x_fe['HouseAge']

x_fe['Pop_Latitude'] = x_fe['Population'] * x_fe['Latitude']

x_fe_train,x_fe_test,y_train,y_test = train_test_split(x_fe,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_fe_train_scaled = scaler.fit_transform(x_fe_train)
x_fe_test_scaled = scaler.transform(x_fe_test)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(x_fe_train_scaled, y_train)
ridge_preds = ridge_model.predict(x_fe_test_scaled)

ridge_mae = mean_absolute_error(y_test, ridge_preds)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
ridge_r2 = r2_score(y_test, ridge_preds)

print("Ridge Regression((TUNED and FEATURED:")
print(f"MAE: {ridge_mae:.4f}, RMSE: {ridge_rmse:.4f}, R²: {ridge_r2:.4f}")

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_fe_train_scaled, y_train)
lasso_preds = lasso_model.predict(x_fe_test_scaled)

lasso_mae = mean_absolute_error(y_test, lasso_preds)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_preds))
lasso_r2 = r2_score(y_test, lasso_preds)

print("\nLasso Regression (TUNED and FEATURED):")
print(f"MAE: {lasso_mae:.4f}, RMSE: {lasso_rmse:.4f}, R²: {lasso_r2:.4f}")


