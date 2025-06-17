import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Project goal: Predicting the price of each house. Building models and finding the best one.")

test = pd.read_csv("D:/ProjectsKaggle/house/test.csv")
train = pd.read_csv("D:/ProjectsKaggle/house/train.csv")
sample = pd.read_csv("D:/ProjectsKaggle/house/sample_submission.csv")
pd.set_option('display.max_columns', None)

print("Number of rows and columns:")
print(train.shape)
print("All rows and columns")
train.info()

print("Descriptive statistics for quantitative columns for train")
print(train.describe())
print("All missing values ​​in columns")
print("We create a DF that has 3 columns. The first is the number of missing values, the second is the percentage of missing values, and the third is the data types.")
print("missing_values - Shows the number of missing values")
print("missing_percentage - Shows in percentage how many values are missing in a column")
print("missing_data - creating DF")
missing_values = train.isnull().sum().sort_values(ascending=False)
missing_percentage = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)
missing_type = train.dtypes
missing_data = pd.concat([missing_values, missing_percentage, missing_type], axis=1, keys=["Total Missing for train", "Percentage for train", "Type for train"])
print("Missing Value Columns for train: ")
print(missing_data[missing_data["Total Missing for train"] > 0])

print("Descriptive statistics for quantitative columns for test")
print(test.describe())
missing_values_test = test.isnull().sum().sort_values(ascending=False)
missing_percentage_test = (test.isnull().sum() / len(test) * 100).sort_values(ascending=False)
missing_type_test = test.dtypes
missing_data_test = pd.concat([missing_values_test, missing_percentage_test, missing_type_test], axis=1, keys=["Total Missing for test", "Percentage for test", "Type for test"])
print("Missing Value Columns for test: ")
print(missing_data_test[missing_data_test["Total Missing for test"] > 0])

print("Replacing empty values ​​with mode and mean values ​​for train")
mode_columns_train = ["GarageQual", "GarageFinish", "GarageType", "GarageCond",
                "BsmtFinType2", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "Electrical"]
for column in mode_columns_train:
    train[column] = train[column].fillna(train[column].mode()[0])

mean_columns_train = ["MasVnrArea", "GarageYrBlt"]
for column in mean_columns_train:
    train[column] = train[column].fillna(train[column].mean())

print("Saving statistics from train")
train_modes = {col: train[col].mode()[0] for col in mode_columns_train}
train_means = {col: train[col].mean() for col in mean_columns_train}

print("Replacing empty values with mode and mean values for test")
mode_columns_test = ["GarageQual", "GarageFinish", "GarageType", "GarageCond", "Utilities", "Functional", "Exterior1st", "Exterior2nd",
                "BsmtFinType2", "BsmtExposure", "BsmtCond", "BsmtQual", "BsmtFinType1", "MSZoning", "SaleType", "KitchenQual"]
for column in mode_columns_test:
    if column in test.columns:
        test[column] = test[column].fillna(train_modes.get(column, test[column].mode()[0]))
print("For test we use the mean value or mode, where the gaps are 10%<x<25%")
mean_columns_test = ["MasVnrArea", "GarageYrBlt", "BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "TotalBsmtSF", "BsmtUnfSF", "BsmtFinSF2",
                     "GarageCars", "GarageArea", "LotFrontage"]
for column in mean_columns_test:
    test[column] = train[column].fillna(train[column].mean())
for column in mean_columns_test:
    if column in test.columns:
        fill_value = train_means.get(column, test[column].mean())
        test[column] = test[column].fillna(fill_value)

print("Using interpolation to replace empty values where gaps are 10%<x<25%")
train["LotFrontage"] = train["LotFrontage"].interpolate()


print("Removing empty values that have a gap value greater than 45% for train")
drop_columns_train = ["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu"]
train = train.drop(columns=drop_columns_train)

print("Removing empty values that have a skip value greater than 45% for test")
drop_columns_test = ["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu"]
test = test.drop(columns=drop_columns_test)

print("Let's see if there are any duplicate lines for train")
print(train.duplicated().sum())

print("Let's see if there are any duplicate lines for test")
print(test.duplicated().sum())

print("Statistics for SalePrice")
print(train["SalePrice"].describe())

print("Graph without normalization")
plt.figure(figsize=(10, 6))
sns.histplot(train['SalePrice'], kde=True, bins=50)
plt.title("SalePrice Scatter")
plt.xlabel("Price (USD)")
plt.ylabel("Frequency")
plt.show()
print("Normalization. Logarithmization.")
train["SalePrice_log"] = np.log1p(train["SalePrice"])
plt.figure(figsize=(10, 6))
sns.histplot(train['SalePrice_log'], kde=True, bins=50, color='green')
plt.title("Log-Transformed SalePrice Scatter")
plt.xlabel("Log(Price)")
plt.ylabel("Frequency")
plt.show()
print("After normalization, it can be seen that the data is centered, which will improve the performance of the model.")

print("Creating a heat map to find correlation")
numerical_cols = train.select_dtypes(include=["int64", "float64"]).columns.tolist()
print(numerical_cols)

corr_matrix = train[numerical_cols].corr()
k = 12
corr_top15 = corr_matrix.nlargest(k, "SalePrice")["SalePrice"].index
cm = np.corrcoef(train[corr_top15].values.T)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, square=True, fmt=".2f",
            annot_kws={"size": 10}, yticklabels=corr_top15.values, xticklabels=corr_top15.values,
            cmap="viridis")
plt.title("SalePrice Most сorralated Feature")
plt.show()

print("Identifying outliers for train")
print("First, you need to plot a graph and make sure that there are a lot of emissions and they are not necessary. For example, a house can really be expensive.")
plt.figure(figsize = (10,6))
sns.boxplot(x = train["OverallQual"], color = "green")
plt.title("OverallQual with outliers")
plt.show()
plt.figure(figsize = (10,6))
sns.boxplot(x = train["GrLivArea"], color = "green")
plt.title("GrLivArea with outliers")
plt.show()
plt.figure(figsize = (10,6))
sns.boxplot(x = train["GarageArea"], color = "green")
plt.title("GarageArea with outliers")
plt.show()

print("Identifying outliers for test")
plt.figure(figsize = (10,6))
sns.boxplot(x = test["OverallQual"], color = "green")
plt.title("OverallQual with outliers")
plt.show()
plt.figure(figsize = (10,6))
sns.boxplot(x = test["GrLivArea"], color = "green")
plt.title("GrLivArea with outliers")
plt.show()
plt.figure(figsize = (10,6))
sns.boxplot(x = test["GarageArea"], color = "green")
plt.title("GarageArea with outliers")
plt.show()

print("Histogram before outliers for GrLivArea for train")
sns.histplot(train['GrLivArea'], kde=True, color='orange')
plt.title("Before Handling Outliers for train")
plt.show()
print("Histogram before outliers for GarageArea for train")
sns.histplot(train['GarageArea'], kde=True, color='orange')
plt.title("Before Handling Outliers for train")
plt.show()

print("Histogram before outliers for GrLivArea for test")
sns.histplot(test['GrLivArea'], kde=True, color='orange')
plt.title("Before Handling Outliers for test")
#plt.show()
print("Histogram before outliers for GarageArea for test")
sns.histplot(test['GarageArea'], kde=True, color='orange')
plt.title("Before Handling Outliers for test")
plt.show()

print("IQR method for data that has many outliers for train")
print("If the values go beyond lower_bound = Q1 - 1.5 * IQR and upper_bound = Q3 + 1.5 * IQR - these are outliers")

Q1 = train["GrLivArea"].quantile(0.25)
Q3 = train["GrLivArea"].quantile(0.75)
IQR = Q3 - Q1
train = train[(train["GrLivArea"] >= Q1 - 1.5*IQR) & (train["GrLivArea"] <= Q3 + 1.5*IQR)]

Q1 = train["GarageArea"].quantile(0.25)
Q3 = train["GarageArea"].quantile(0.75)
IQR = Q3 - Q1
train = train[(train["GarageArea"] >= Q1 - 1.5*IQR) & (train["GarageArea"] <= Q3 + 1.5*IQR)]

print("IQR method for data that have many outliers for test")
test["GrLivArea"] = test["GrLivArea"].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

print("Resetting indices for train")
train = train.reset_index(drop=True)

print("Resetting indices for test")
test = test.reset_index(drop=True)

print("Histogram without outliers for GrLivArea for train")
sns.histplot(train["GrLivArea"], kde=True, color='red')
plt.title("After Handling Outliers for train")
plt.show()

print("Outlier-free histogram for GarageArea for train")
sns.histplot(train["GarageArea"], kde=True, color='red')
plt.title("After Handling Outliers for train")
plt.show()

print("Histogram without outliers for GrLivArea for test")
sns.histplot(test["GrLivArea"], kde=True, color='red')
plt.title("After Handling Outliers for test")
plt.show()

print("Histogram without outliers for GarageArea for test")
sns.histplot(test["GarageArea"], kde=True, color='red')
plt.title("After Handling Outliers for test")
plt.show()

print("Using Categorical Data for Analysis. Neighborhood for train")
print("Data processing")
train["Neighborhood"] = train["Neighborhood"].str.lower().str.strip()
print("One-hot encoding")
df_encoded = pd.get_dummies(train, columns=["Neighborhood"], drop_first=True)
neighborhood_columns = [col for col in df_encoded.columns if col.startswith('Neighborhood_')]

print(df_encoded.head())

print("Using Categorical Data for Analysis. Neighborhood for test")
test["Neighborhood"] = test["Neighborhood"].str.lower().str.strip()
print("One-hot encoding")
df_encoded_t = pd.get_dummies(test, columns=["Neighborhood"], drop_first=True)
for col in neighborhood_columns:
    if col not in df_encoded_t.columns:
        df_encoded_t[col] = 0
print(df_encoded_t.head())

print("In which area are houses more expensive?")
plt.figure(figsize=(12,6))
sns.boxplot(x="Neighborhood", y='SalePrice', data=train)
plt.xticks(rotation=45)
plt.title("SalePrice by Neighborhood")
plt.show()

print("Creating a new df for the model for train")
numeric_features = train[["OverallQual", "GrLivArea", "GarageArea"]]
neighborhood_features = df_encoded.filter(regex='^Neighborhood_')
all_features = pd.concat([numeric_features, neighborhood_features], axis=1)
print(all_features.head())

print("Creating a new df for the model for test")
numeric_features_t = test[["OverallQual", "GrLivArea", "GarageArea"]]
neighborhood_features_t = df_encoded_t.filter(regex='^Neighborhood_')
all_features_t = pd.concat([numeric_features_t, neighborhood_features_t], axis=1)
print(all_features_t.head())

print("Column alignment")
all_features_t = all_features_t[all_features.columns]
print(all_features_t.head())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(all_features, train["SalePrice_log"], test_size = 0.2, random_state = 42)

print("Model training")
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
print("Linear")
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
print("RandomForestRegressor")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print("Metrics for linear regression. Log")
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)
print("RMSE", rmse_lr)
print("R2", r2_lr)

print("Metrics for RandomForestRegressor. Log")
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print("RMSE", rmse_rf)
print("R2", r2_rf)

y_test_original = np.expm1(y_test)

print("Metrics for linear regression.")
y_pred_lr_original = np.expm1(y_pred_lr)
rmse_lr_original = np.sqrt(mean_squared_error(y_test_original, y_pred_lr_original))
r2_lr_original = r2_score(y_test_original, y_pred_lr_original)
print("RMSE", rmse_lr_original)
print("R2", r2_lr_original)

print("Metrics for RandomForestRegressor.")
y_pred_rf_original = np.expm1(y_pred_rf)
rmse_rf_original = np.sqrt(mean_squared_error(y_test_original, y_pred_rf_original))
r2_rf_original = r2_score(y_test_original, y_pred_rf_original)
print("RMSE", rmse_rf_original)
print("R2", r2_rf_original)

print("Cross-validation. Mean. Standard deviation. Linear model")
from sklearn.model_selection import cross_val_score
scores_lr = cross_val_score(lr, all_features, train["SalePrice_log"], cv = 3)
mean_lr = scores_lr.mean()
std_lr = scores_lr.std()
print("Average RMSE", mean_lr, "Standard Deviation", std_lr)

print("Cross-validation. Average. Standard Deviation. Random Forest")
scores_rf = cross_val_score(rf, all_features, train["SalePrice_log"], cv=3)
rmse_scores_rf = -scores_rf
mean_rf = rmse_scores_rf.mean()
std_rf = rmse_scores_rf.std()
print("Average RMSE:" , mean_rf, "Standard Deviation", std_rf)

print("Comparison of models")
models = ["Linear Regression", "Random Forest"]
rmse_values = [rmse_lr_original, rmse_rf_original]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=rmse_values, palette="viridis")
plt.title("Comparison of models by RMSE")
plt.ylabel("RMSE (original scale)")
plt.show()

print("Train the best model on all data")
best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(all_features, train["SalePrice_log"])

print("Predictions for test data")
test_predictions_log = best_model.predict(all_features_t)
test_predictions = np.expm1(test_predictions_log)

print("Creating a file")
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": test_predictions
})
submission.to_csv('D:/ProjectsKaggle/house/submission.csv', index=False)

print("Visualization of Predictions")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(train["SalePrice"], bins=50, kde=True, color="blue")
plt.title("Actual prices(train)")

plt.subplot(1, 2, 2)
sns.histplot(submission["SalePrice"], bins=50, kde=True, color="red")
plt.title("Predicted prices(test)")

plt.tight_layout()
plt.show()
