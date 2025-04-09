from sklearn.model_selection import GridSearchCV

# Define alpha values to try
alphas = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Ridge
ridge_gs = GridSearchCV(Ridge(), alphas, cv=5, scoring='neg_mean_squared_error')
ridge_gs.fit(X_train_scaled, y_train)
best_ridge = ridge_gs.best_estimator_

# Lasso
lasso_gs = GridSearchCV(Lasso(), alphas, cv=5, scoring='neg_mean_squared_error')
lasso_gs.fit(X_train_scaled, y_train)
best_lasso = lasso_gs.best_estimator_

# Evaluate


ridge_preds = best_ridge.predict(X_test_scaled)
lasso_preds = best_lasso.predict(X_test_scaled)

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
