from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('instagram_usage_lifestyle.csv')

features = [
    'age',    'daily_active_minutes_instagram',
    'self_reported_happiness'
]

target = ['perceived_stress_score']

scaler = StandardScaler()
minmax = MinMaxScaler(feature_range=(1,10))

df = df[df['country'] == 'United States']

df = df.head(50000)
# df['active_consumption'] = (df['dms_sent_per_week']/7 + df['comments_written_per_day'] + df['posts_created_per_week']) / df['daily_active_minutes_instagram']
# df['passive_consumption'] = (df['reels_watched_per_day'] + df['stories_viewed_per_day']) / df['daily_active_minutes_instagram']
# df['active_ratio'] = df['active_consumption'] / ( df['passive_consumption'] + df['active_consumption'])
data = df[['age',
           'daily_active_minutes_instagram',
'self_reported_happiness', 'perceived_stress_score']]

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

X_train_nonpoly = scaler.fit_transform(X_train)
X_test_nonpoly = scaler.transform(X_test)
y_train = minmax.fit_transform(y_train)
y_test = minmax.transform(y_test)

# Poly regression model
poly = PolynomialFeatures(degree = 3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_test_poly = scaler.transform(X_test_poly)

# Polynomial with Ridge:
model = Ridge(alpha = 1.5, fit_intercept = True)
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

training_mse = mean_squared_error(y_train, model.predict(X_train_poly), multioutput='raw_values')
training_r2 = r2_score(y_train, model.predict(X_train_poly), multioutput='raw_values')
print("Ridge Polynomial Results:")
print("training:")
print(f"Stress MSE {training_mse[0]} and Stress R2 = {training_r2[0]}")
print(f"RMSE: {np.sqrt(training_mse[0])}")

testing_mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
testing_r2 = r2_score(y_test, y_pred, multioutput='raw_values')
print("Testing")
print(f"Stress MSE {testing_mse[0]} and Stress R2 = {testing_r2[0]}")
print(f"RMSE: {np.sqrt(testing_mse[0])}")

rfmodel = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8, min_samples_leaf=8, min_samples_split=20)
rfmodel.fit(X_train_nonpoly, y_train)
y_pred_rf = rfmodel.predict(X_test_nonpoly)

training_mse_rf = mean_squared_error(y_train, rfmodel.predict(X_train_nonpoly), multioutput='raw_values')
training_r2_rf = r2_score(y_train, rfmodel.predict(X_train_nonpoly), multioutput='raw_values')
print("Random Forest training stats:")
print(f"Stress MSE {training_mse_rf[0]} and Stress R2 = {training_r2_rf[0]}")
print(f"RMSE: {np.sqrt(training_mse_rf[0])}")

mse_rf = mean_squared_error(y_test, y_pred_rf, multioutput='raw_values')
r2_rf = r2_score(y_test, y_pred_rf, multioutput='raw_values')
print("Random Forest test stats:")
print(f"Stress MSE {mse_rf[0]} and Stress R2 = {r2_rf[0]}")
print(f"RMSE: {np.sqrt(mse_rf[0])}")

xgb_estimator = xgb.XGBRegressor(objective='reg:squarederror', booster='dart', learning_rate = 0.1, max_depth = 5, min_child_weight = 3, n_estimators = 100, rate_drop = 0.1, reg_alpha = 0, reg_lambda = 1, subsample = 1.0, random_state=42)
xgb_estimator.fit(X_train_nonpoly, y_train)

y_train_pred = xgb_estimator.predict(X_train_nonpoly)
y_test_pred = xgb_estimator.predict(X_test_nonpoly)

print("Stress Level Predictions")
print("Training Data:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred))}")
print(f"MSE: {mean_squared_error(y_train, y_train_pred)}")
print(f"R² Score: {r2_score(y_train, y_train_pred)}\n")

print("Test Data:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred))}")
print(f"MSE: {mean_squared_error(y_test, y_test_pred)}")
print(f"R² Score: {r2_score(y_test, y_test_pred)}\n")

joblib.dump(model, 'poly_regression.joblib')
joblib.dump(rfmodel, 'randomforest.joblib')
joblib.dump(xgb_estimator, 'xgboost.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(poly, 'poly.joblib')