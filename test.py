import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("/Users/clarencemarvin/Downloads/Health_insurance.csv")

print(data.isnull().sum()) #check if there are any null variable

data["sex"] = data["sex"].map({"female": 0, "male": 1})
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1}) #convert the data from caterogical to numerical

X = data[["age", "sex", "bmi", "smoker"]]
y = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

forest = RandomForestRegressor(random_state=42) #initialize random forest

param_grid = {
    'n_estimators': [50, 100, 200], #trying few models of random forest
    'max_depth': [None, 10, 20, 30], #limit the growth of tree to prevent overfitting
    'min_samples_split': [2, 5, 10] #limit the minimum size to prevent underfitting or overfitting
}

grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_forest = grid_search.best_estimator_

y_pred = best_forest.predict(X_test_scaled)

data = pd.DataFrame(data={"\nPredicted Premium Amount": y_pred})
print(data.head())
print(f"\nR² Score: {r2_score(y_test, y_pred)}")

importances = best_forest.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame(sorted(zip(importances, feature_names), reverse=True), columns=['Importance', 'Feature'])
print("\nFeature Importances:")
print(feature_importances)