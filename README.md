# Health Insurance Cost Prediction

## Project Overview
This project aims to predict individual medical costs billed by health insurance using a dataset of insurance beneficiaries. We employ a RandomForestRegressor, a powerful ensemble learning method, to predict the insurance charges based on features like age, sex, body mass index, and smoking status. The project includes data preprocessing, model training with hyperparameter tuning using GridSearchCV, and evaluation of the best model's predictive performance.

## Dataset
The dataset used in this project is `Health_insurance.csv`, which contains several attributes related to insurance beneficiaries. The attributes include age, sex, BMI (Body Mass Index), smoking status, and insurance charges.

## Data Preprocessing
The dataset underwent several preprocessing steps:
  1. Null value check and handling.
  2. Conversion of categorical variables (sex and smoker) to numerical form.
  3. Feature scaling using StandardScaler to standardize the feature set.

## Model Training and Hyperparameter Tuning
The project utilizes GridSearchCV to perform hyperparameter tuning on a RandomForestRegressor. The parameters tuned include the number of estimators, maximum depth of the trees, and the minimum number of samples required to split a node.

## Results
  1. R² Score: 0.8584517595319375
  2. Predicted Premium Amounts:
     <img width="229" alt="Screenshot 2024-02-19 at 9 41 16 PM" src="https://github.com/clarencemarvin/Insurance_Premium_Predictor/assets/124359735/eba110b5-a9e5-4c6d-897a-9ebd3d2da85f">
     
  3. Correlation Map:
     <img width="192" alt="Screenshot 2024-02-19 at 9 40 47 PM" src="https://github.com/clarencemarvin/Insurance_Premium_Predictor/assets/124359735/674a55ee-d4a2-414b-940c-87e75fcf7d83">


## Conclusion
The Random Forest Regressor model with hyperparameters in the project demonstrates a strong ability to predict health insurance costs, as reflected in the high R² score. This indicates that the model can be a valuable tool in understanding and estimating insurance charges based on demographic and health-related characteristics of beneficiaries. The feature importances highlight which features are most influential in predicting the charges, providing insights into the factors that most significantly impact health insurance costs.
