import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

data = fetch_california_housing(as_frame=True)
df = data.frame 

df_clean = df.copy()

df_clean = df_clean[df_clean['AveOccup']< 100]


x = df_clean.drop("MedHouseVal", axis=1)
y = df_clean["MedHouseVal"]

# Define alpha values to try
alphas = {'alpha': [0.01, 0.1, 1, 10, 100]}


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("X_train shape : ",x_train_scaled.shape)

print("Example scaled features (first row): ", x_train_scaled[0])


# Ridge
ridge_gs = GridSearchCV(Ridge(), alphas, cv=5, scoring='neg_mean_squared_error')
ridge_gs.fit(x_train_scaled, y_train)
best_ridge = ridge_gs.best_estimator_

# Lasso
lasso_gs = GridSearchCV(Lasso(), alphas, cv=5, scoring='neg_mean_squared_error')
lasso_gs.fit(x_train_scaled, y_train)
best_lasso = lasso_gs.best_estimator_

# Evaluate


ridge_preds = best_ridge.predict(x_test_scaled)
lasso_preds = best_lasso.predict(x_test_scaled)

print("Best Ridge Alpha:", ridge_gs.best_params_)
print("Best Lasso Alpha:", lasso_gs.best_params_)

print("\nRidge:")
print(f"MAE: {mean_absolute_error(y_test, ridge_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, ridge_preds)):.4f}")
print(f"R²: {r2_score(y_test, ridge_preds):.4f}")

print("\nLasso:")
print(f"MAE: {mean_absolute_error(y_test, lasso_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, lasso_preds)):.4f}")
print(f"R²: {r2_score(y_test, lasso_preds):.4f}")
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

print("\nLasso Regression (TUNED):")
print(f"MAE: {lasso_mae:.4f}, RMSE: {lasso_rmse:.4f}, R²: {lasso_r2:.4f}")
print(" ")



