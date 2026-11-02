# EX-NO.-4a-MACHINE-LEARNING-MODEL-LINEAR-REGRESSION
## AIM
To predict house prices using regression models and compare the performance of different machine learning regression models based on RMSE, MAE, and R².
1.Machine Learning:Machine Learning is used to learn patterns from existing data and make predictions. 
•	Regression is a supervised learning technique used to predict continuous numerical values. 
•	In this experiment, regression models are used to predict the price of a house. 
•	The dataset contains house-related features such as: 
o	square_feet 
o	num_rooms 
o	age 
o	distance_to_city(km) 
•	The target variable is: 
o	price 
## DATASET DESCRIPTION
•	Dataset: House Price Dataset 
•	Problem: Predict house price. 
•	Features (X): 
o	square_feet – size of the house. 
o	num_rooms – number of rooms. 
o	age – age of the house in years. 
o	distance_to_city(km) – distance from the city centre. 
•	Target (y): 
o	price – continuous house price. 
## PROBLEM STATEMENT
•	Develop a machine learning model to predict house prices. 
•	Use house characteristics as input. 
•	Train different regression models. 
•	Compare their prediction performance. 
•	Select the better-performing model based on evaluation metrics. 
## REGRESSION MODELS USED
The uploaded notebook compares the following models:
1.	Linear Regression 
2.	Ridge Regression 
3.	Lasso Regression 
4.	ElasticNet Regression 
5.	Polynomial Regression 
6.	Decision Tree Regressor 
7.	Random Forest Regressor 
8.	Gradient Boosting Regressor 
9.	Support Vector Regressor (SVR) 
10.	K-Nearest Neighbors (KNN) Regressor 
### LIBRARIES USED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
### LOAD THE DATASET
import pandas as pd

df = pd.read_csv(
    '/content/drive/MyDrive/Datasets/house_prices_dataset.csv'
)

df.head()
### DATA OVERVIEW
Display the dataset
df
Display information
df.info()
Display shape
df.shape
Summary statistics
df.describe()
Check missing values
df.isnull().sum()
## EXPLORATORY DATA ANALYSIS
Distribution of Features
The notebook examines the distribution of:
•	square_feet 
•	num_rooms 
•	age 
•	distance_to_city(km) 
•	price 
numeric_features = [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()
9. CORRELATION ANALYSIS
•	Correlation shows the relationship between numerical variables. 
•	A correlation heatmap is used to visualize these relationships. 
plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()
10. SCATTER PLOTS
Scatter plots are used to study the relationship between individual features and house price.
for col in [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)'
]:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[col], y=df['price'])
    plt.title(f'{col} vs Price')
    plt.show()
## OUTLIER DETECTION
•	Boxplots are used to identify extreme values. 
•	Outliers may negatively affect regression models. 
for col in [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]:
    plt.figure(figsize=(6,4))
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
## OUTLIER TREATMENT
The notebook removes extremely low and extremely high house prices using the 1st and 99th percentiles.
Q1 = df['price'].quantile(0.01)
Q99 = df['price'].quantile(0.99)

df = df[
    (df['price'] >= Q1) &
    (df['price'] <= Q99)
]
•	This reduces the effect of extreme house prices. 
•	It helps the models learn from more typical observations. 
13. DEFINE FEATURES AND TARGET
X = df[
    [
        'square_feet',
        'num_rooms',
        'age',
        'distance_to_city(km)'
    ]
]

y = df['price']
•	X → Input features. 
•	y → Target house price. 
14. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
•	80% → Training data 
•	20% → Testing data 
## FEATURE SCALING
The notebook uses StandardScaler.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
•	Scaling puts the features on a comparable scale. 
•	The scaler is fitted only on training data. 
•	The same transformation is applied to test data. 

## BASELINE MODEL
The baseline predicts the mean house price for every test observation.
y_pred_baseline = (
    np.mean(y_train) *
    np.ones_like(y_test)
)

rmse_baseline = np.sqrt(
    mean_squared_error(y_test, y_pred_baseline)
)

mae_baseline = mean_absolute_error(
    y_test,
    y_pred_baseline
)

print(
    f"Baseline RMSE: {rmse_baseline:.2f}, "
    f"MAE: {mae_baseline:.2f}"
)

## LINEAR REGRESSION
•	Linear Regression finds a linear relationship between input features and house price. 
•	It is used as the main baseline regression model. 
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

## RIDGE REGRESSION
•	Ridge Regression is a regularized version of Linear Regression. 
•	It helps control large model coefficients. 
ridge = Ridge(alpha=1.0)

ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)

## LASSO REGRESSION
•	Lasso Regression uses L1 regularization. 
•	It can reduce some feature coefficients toward zero. 
lasso = Lasso(alpha=0.1)

lasso.fit(X_train_scaled, y_train)

y_pred_lasso = lasso.predict(X_test_scaled)

##  ELASTIC NET REGRESSION
•	ElasticNet combines L1 and L2 regularization. 
•	It is useful when several features may contribute to the prediction. 
elastic = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

elastic.fit(X_train_scaled, y_train)

y_pred_elastic = elastic.predict(X_test_scaled)

##  POLYNOMIAL REGRESSION
•	Polynomial Regression extends Linear Regression by creating polynomial features. 
•	The notebook uses degree 2. 
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_lr = LinearRegression()

poly_lr.fit(X_train_poly, y_train)

y_pred_poly = poly_lr.predict(X_test_poly)

##  DECISION TREE REGRESSOR
•	A Decision Tree divides the data into different regions based on feature values. 
•	It can model nonlinear relationships. 
dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train_scaled, y_train)

y_pred_dt = dt.predict(X_test_scaled)

##  RANDOM FOREST REGRESSOR
•	Random Forest combines multiple decision trees. 
•	It generally provides more robust predictions than a single tree. 
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

##  GRADIENT BOOSTING REGRESSOR
•	Gradient Boosting builds models sequentially. 
•	Each new model attempts to improve the errors of previous models. 
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gbr.fit(X_train_scaled, y_train)

y_pred_gbr = gbr.predict(X_test_scaled)

## SUPPORT VECTOR REGRESSOR
•	SVR uses Support Vector Machine principles for regression. 
•	The notebook uses an RBF kernel. 
svr = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=.1
)

svr.fit(X_train_scaled, y_train)

y_pred_svr = svr.predict(X_test_scaled)

##  KNN REGRESSOR
•	KNN predicts a value based on nearby observations. 
•	The notebook uses 5 neighbors. 
knn = KNeighborsRegressor(
    n_neighbors=5
)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

## MODEL EVALUATION
The notebook uses three metrics:
RMSE
•	Root Mean Squared Error. 
•	Lower value indicates better performance. 
MAE
•	Mean Absolute Error. 
•	Lower value indicates better performance. 
R²
•	Measures how well the model explains variation in house prices. 
•	Higher value generally indicates better performance. 
Code
models = {
    "Linear Regression": y_pred_lr,
    "Ridge": y_pred_ridge,
    "Lasso": y_pred_lasso,
    "ElasticNet": y_pred_elastic,
    "Polynomial Regression": y_pred_poly,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "Gradient Boosting": y_pred_gbr,
    "SVR": y_pred_svr,
    "KNN": y_pred_knn
}

results = []

for name, y_pred in models.items():

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append([
        name,
        rmse,
        mae,
        r2
    ])

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

results_df.sort_values(
    by="RMSE"
)
		

### Comparison Criteria
•	Lower RMSE → Better model. 
•	Lower MAE → Better model. 
•	Higher R² → Better model. 
The notebook sorts the models according to RMSE to compare their performance.
## ACTUAL VS PREDICTED PRICE
plt.figure(figsize=(15,12))

for i, (name, y_pred) in enumerate(models.items()):

    plt.subplot(5,2,i+1)

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name}: Actual vs Predicted")

plt.tight_layout()
plt.show()
•	The plot compares actual house prices with predicted prices. 
•	Points closer to the diagonal line indicate better predictions. 
30. RESIDUAL ANALYSIS
•	A residual is the difference between actual and predicted values. 
Residual = Actual Price − Predicted Price
for name, y_pred in models.items():

    residuals = y_test - y_pred

    plt.figure(figsize=(6,4))

    sns.scatterplot(
        x=y_pred,
        y=residuals,
        alpha=0.5
    )

    plt.axhline(
        0,
        color='r',
        linestyle='--'
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title(f"{name}: Residual Plot")

    plt.show()
•	Residuals close to zero indicate smaller prediction errors. 
•	Residual plots help identify unusual prediction patterns. 
##  RANDOM FOREST FEATURE IMPORTANCE
The notebook calculates the importance of each feature using Random Forest.
importances = rf.feature_importances_

feat_names = X.columns

plt.figure(figsize=(6,4))

sns.barplot(
    x=importances,
    y=feat_names
)

plt.title(
    "Random Forest Feature Importance"
)

plt.show()
This helps identify which house features contribute more to the Random Forest prediction.
##  GRADIENT BOOSTING FEATURE IMPORTANCE
importances_gbr = gbr.feature_importances_

plt.figure(figsize=(6,4))

sns.barplot(
    x=importances_gbr,
    y=feat_names
)

plt.title(  "Gradient Boosting Feature Importance")

plt.show()
This shows the relative contribution of the house features to the Gradient Boosting model.

## RMSE COMPARISON GRAPH
plt.figure(figsize=(10,6))

sns.barplot(
    x="RMSE",
    y="Model",
    data=results_df.sort_values("RMSE")
)

plt.title(
    "RMSE Comparison Across Regression Models"
)

plt.show()
•	The graph provides a visual comparison of model errors. 
•	The model with the lowest RMSE performs best according to this metric.

# PROGRAM   
````````````````
from google.colab import drive

drive.mount('/content/drive')

import pandas as pd

df = pd.read_csv('/content/drive/My Drive/house_datasets.csv')

df.head()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Dataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

sns.heatmap(
df.corr(numeric_only=True),
annot=True,
cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

features = [
'square_feet',
'num_rooms',
'age',
'distance_to_city(km)'
 ]

for feature in features:

plt.figure(figsize=(6, 4))

plt.scatter(
df[feature],
df['price']
)

plt.xlabel(feature)
plt.ylabel('Price')
plt.title(feature + ' vs Price')

plt.show()

plt.figure(figsize=(6, 4))

sns.boxplot(
y=df['price']
)

plt.title("Price Outliers")
plt.show()



lower_limit = df['price'].quantile(0.01)
upper_limit = df['price'].quantile(0.99)

df = df[
(df['price'] >= lower_limit) &
(df['price'] <= upper_limit)
]

print("Shape after removing outliers:")
print(df.shape)



X = df[
[
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)'
]
]

y = df['price']

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())



X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed")



baseline_prediction = np.full(
len(y_test),
y_train.mean()
)

baseline_rmse = np.sqrt(
mean_squared_error(
    y_test,
    baseline_prediction
)
)

baseline_mae = mean_absolute_error(
y_test,
baseline_prediction
)

baseline_r2 = r2_score(
y_test,
baseline_prediction
)

print("\nBaseline Model")
print("RMSE:", baseline_rmse)
print("MAE:", baseline_mae)
print("R2:", baseline_r2)



linear_model = LinearRegression()

linear_model.fit(
X_train_scaled,
y_train
)

linear_pred = linear_model.predict(
X_test_scaled
)


ridge_model = Ridge()

ridge_model.fit(
X_train_scaled,
y_train
)

ridge_pred = ridge_model.predict(
X_test_scaled
)



lasso_model = Lasso()

lasso_model.fit(
X_train_scaled,
y_train
)

lasso_pred = lasso_model.predict(
X_test_scaled
)


elastic_model = ElasticNet()

elastic_model.fit(
X_train_scaled,
y_train
)

elastic_pred = elastic_model.predict(
X_test_scaled
)



poly = PolynomialFeatures(
degree=2
)

X_train_poly = poly.fit_transform(
X_train_scaled
)

X_test_poly = poly.transform(
X_test_scaled
)

poly_model = LinearRegression()

poly_model.fit(
X_train_poly,
y_train
)

poly_pred = poly_model.predict(
X_test_poly
)


dt_model = DecisionTreeRegressor(
random_state=42
)

dt_model.fit(
X_train,
y_train
)

dt_pred = dt_model.predict(
X_test
)



rf_model = RandomForestRegressor(
n_estimators=100,
random_state=42
)

rf_model.fit(
X_train,
y_train
)

rf_pred = rf_model.predict(
X_test
)



gb_model = GradientBoostingRegressor(
n_estimators=100,
learning_rate=0.1,
random_state=42
)

gb_model.fit(
X_train,
y_train
)

gb_pred = gb_model.predict(
X_test
)



svr_model = SVR(
kernel='rbf',
C=100,
gamma=0.1,
epsilon=0.1
)

svr_model.fit(
X_train_scaled,
y_train
)

svr_pred = svr_model.predict(
X_test_scaled
)



knn_model = KNeighborsRegressor(
n_neighbors=5
)

knn_model.fit(
X_train_scaled,
y_train
)

knn_pred = knn_model.predict(
X_test_scaled
)



def evaluate_model(name, y_true, y_pred):

rmse = np.sqrt(
mean_squared_error(
    y_true,
    y_pred
)
)

mae = mean_absolute_error(
y_true,
y_pred
)

r2 = r2_score(
y_true,
y_pred
)

return {
'Model': name,
'RMSE': rmse,
'MAE': mae,
'R2': r2
}


results = []

results.append(
evaluate_model(
    'Linear Regression',
    y_test,
    linear_pred
)
)

results.append(
evaluate_model(
    'Ridge Regression',
    y_test,
    ridge_pred
)
)

results.append(
evaluate_model(
    'Lasso Regression',
    y_test,
    lasso_pred
)
)

results.append(
evaluate_model(
    'ElasticNet',
    y_test,
    elastic_pred
)
)

results.append(
evaluate_model(
    'Polynomial Regression',
    y_test,
    poly_pred
)
)

results.append(
evaluate_model(
    'Decision Tree',
y_test,
dt_pred
)
)

results.append(
evaluate_model(
    'Random Forest',
    y_test,
    rf_pred
)
)

results.append(
evaluate_model(
    'Gradient Boosting',
    y_test,
    gb_pred
)
)

results.append(
evaluate_model(
    'SVR',
    y_test,
    svr_pred
)
)

results.append(
evaluate_model(
'KNN',
y_test,
knn_pred
)
)


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
by='RMSE'
 )

print("\nMODEL COMPARISON")
print(results_df)



predictions = {
'Linear Regression': linear_pred,
'Ridge': ridge_pred,
'Lasso': lasso_pred,
'ElasticNet': elastic_pred,
'Polynomial': poly_pred,
'Decision Tree': dt_pred,
'Random Forest': rf_pred,
'Gradient Boosting': gb_pred,
'SVR': svr_pred,
'KNN': knn_pred
}

for name, prediction in predictions.items():

plt.figure(figsize=(6, 4))

plt.scatter(
y_test,
prediction
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title(
name + " - Actual vs Predicted"
 )

plt.show()


for name, prediction in predictions.items():

residuals = y_test - prediction

plt.figure(figsize=(6, 4))

plt.scatter(
prediction,
residuals
)

 plt.axhline(
y=0,
linestyle='--'
)

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")

plt.title(
name + " - Residual Plot"
)

plt.show()



importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
'Feature': X.columns,
'Importance': importance
})

feature_importance = feature_importance.sort_values(
by='Importance',
ascending=False
)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

plt.figure(figsize=(8, 5))

plt.bar(
feature_importance['Feature'],
feature_importance['Importance']
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


gb_importance = gb_model.feature_importances_

gb_feature_importance = pd.DataFrame({
'Feature': X.columns,
'Importance': gb_importance
})

gb_feature_importance = gb_feature_importance.sort_values(
by='Importance',
ascending=False
)

print("\nGradient Boosting Feature Importance:")
print(gb_feature_importance)



plt.figure(figsize=(10, 6))

plt.bar(
results_df['Model'],
results_df['RMSE']
 )

plt.xlabel("Models")
plt.ylabel("RMSE")

plt.title(
"RMSE Comparison of Regression Models"
)

plt.xticks(
rotation=45,
ha='right'
)

plt.tight_layout()
plt.show()
````````````````
# OUTPUT
<img width="705" height="615" alt="WhatsApp Image 2026-11-02 at 10 25 26 AM" src="https://github.com/user-attachments/assets/e1778938-7fe8-4253-8693-ecdbfc0827ce" />
<img width="705" height="615" alt="WhatsApp Image 2026-11-02 at 10 25 26 AM" src="https://github.com/user-attachments/assets/8835c402-a905-44af-b671-e413b2c6ba90" />
<img width="762" height="747" alt="WhatsApp Image 2026-11-02 at 10 25 33 AM" src="https://github.com/user-attachments/assets/470276d8-c6a8-4372-b1d6-397ab1d2a503" />
<img width="762" height="747" alt="WhatsApp Image 2026-11-02 at 10 25 33 AM" src="https://github.com/user-attachments/assets/c7d0d17e-750f-486e-a211-84f08eb1a4b7" />
<img width="725" height="375" alt="WhatsApp Image 2026-11-02 at 10 26 10 AM" src="https://github.com/user-attachments/assets/2360e756-ed60-4347-9f04-eb6dfc7654f6" />
<img width="761" height="682" alt="WhatsApp Image 2026-11-02 at 10 26 17 AM" src="https://github.com/user-attachments/assets/105accbf-7ede-4d04-80c1-c69c54c3832f" />
<img width="726" height="325" alt="WhatsApp Image 2026-11-02 at 10 26 28 AM" src="https://github.com/user-attachments/assets/cf7e286b-3f96-49ad-a8de-1617bcdf40e6" />
<img width="802" height="647" alt="WhatsApp Image 2026-11-02 at 10 26 48 AM" src="https://github.com/user-attachments/assets/6e73e6f6-75c4-488c-b39e-8e38b03fc3e6" />
<img width="722" height="467" alt="WhatsApp Image 2026-11-02 at 10 26 55 AM" src="https://github.com/user-attachments/assets/ca2fecec-737b-40e4-8b9a-162fc6ab8e4f" />
<img width="722" height="467" alt="WhatsApp Image 2026-11-02 at 10 26 55 AM" src="https://github.com/user-attachments/assets/56bd9f15-9359-40d4-b63e-f14a609a4082" />
<img width="787" height="517" alt="WhatsApp Image 2026-11-02 at 10 27 03 AM" src="https://github.com/user-attachments/assets/1bba49d8-e635-4282-a86f-144419524ac1" />
<img width="777" height="482" alt="WhatsApp Image 2026-11-02 at 10 27 28 AM" src="https://github.com/user-attachments/assets/9d05ba25-d501-4969-9781-313375f0bce0" />
<img width="730" height="750" alt="WhatsApp Image 2026-11-02 at 10 27 36 AM" src="https://github.com/user-attachments/assets/9a56ccda-bbe9-4642-820c-8db9ff9e5906" />
<img width="837" height="802" alt="WhatsApp Image 2026-11-02 at 10 27 45 AM" src="https://github.com/user-attachments/assets/2cfee38a-52f8-4213-be3c-f52649896e46" />
<img width="792" height="517" alt="WhatsApp Image 2026-11-02 at 10 27 53 AM" src="https://github.com/user-attachments/assets/e79033af-b236-4a0f-a8cf-68058c28ec55" />
<img width="792" height="517" alt="WhatsApp Image 2026-11-02 at 10 27 53 AM" src="https://github.com/user-attachments/assets/6add87ed-bcb3-48fc-88fc-ce7448f15201" />
<img width="792" height="517" alt="WhatsApp Image 2026-11-02 at 10 27 53 AM" src="https://github.com/user-attachments/assets/1314a8cd-607e-4e30-8345-684304f1aee4" />
<img width="727" height="462" alt="WhatsApp Image 2026-11-02 at 10 28 20 AM" src="https://github.com/user-attachments/assets/984ffaac-5143-496e-80ed-04967b1571c5" />
<img width="722" height="500" alt="WhatsApp Image 2026-11-02 at 10 29 19 AM" src="https://github.com/user-attachments/assets/bcf1c926-df9c-432f-bd45-870a887b8ae4" />
<img width="736" height="477" alt="WhatsApp Image 2026-11-02 at 10 28 50 AM" src="https://github.com/user-attachments/assets/17a25721-9130-4979-bbcf-414bcdeedc01" />
<img width="717" height="481" alt="WhatsApp Image 2026-11-02 at 10 28 57 AM" src="https://github.com/user-attachments/assets/e53eb2c7-ab03-408c-8f8f-6bf611b623b8" />
<img width="711" height="472" alt="WhatsApp Image 2026-11-02 at 10 29 03 AM" src="https://github.com/user-attachments/assets/e8dd1616-86a0-4311-8fd3-07914caa11dd" />
<img width="717" height="497" alt="WhatsApp Image 2026-11-02 at 10 29 52 AM" src="https://github.com/user-attachments/assets/0afb8d23-2e29-4f5e-a458-73f633e7ba61" />
<img width="730" height="487" alt="WhatsApp Image 2026-11-02 at 10 29 59 AM" src="https://github.com/user-attachments/assets/7ece798f-563d-48c2-95bc-67a00ec47522" />
<img width="727" height="472" alt="WhatsApp Image 2026-11-02 at 10 30 05 AM" src="https://github.com/user-attachments/assets/f9ed5b93-6535-4ca7-9958-b3fbc36e1ad2" />
<img width="692" height="510" alt="WhatsApp Image 2026-11-02 at 10 30 11 AM" src="https://github.com/user-attachments/assets/272f7b98-8925-4f70-9c28-93206fd042a5" />

## RESULT
Thus, Linear Regression and other regression models were successfully applied for house price prediction, and their performance was compared using standard regression evaluation metrics.


<img width="711" height="472" alt="WhatsApp Image 2026-11-02 at 10 29 03 AM" src="https://github.com/user-attachments/assets/e8dd1616-86a0-4311-8fd3-07914caa11dd" />
<img width="717" height="497" alt="WhatsApp Image 2026-11-02 at 10 29 52 AM" src="https://github.com/user-attachments/assets/0afb8d23-2e29-4f5e-a458-73f633e7ba61" />
<img width="730" height="487" alt="WhatsApp Image 2026-11-02 at 10 29 59 AM" src="https://github.com/user-attachments/assets/7ece798f-563d-48c2-95bc-67a00ec47522" />
<img width="727" height="472" alt="WhatsApp Image 2026-11-02 at 10 30 05 AM" src="https://github.com/user-attachments/assets/f9ed5b93-6535-4ca7-9958-b3fbc36e1ad2" />
<img width="692" height="510" alt="WhatsApp Image 2026-11-02 at 10 30 11 AM" src="https://github.com/user-attachments/assets/272f7b98-8925-4f70-9c28-93206fd042a5" />
