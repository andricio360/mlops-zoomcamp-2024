import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df_jan = pd.read_parquet("/content/yellow_tripdata_2023-01.parquet")
df_feb = pd.read_parquet("/content/yellow_tripdata_2023-02.parquet")
# printing len of df january columns
print(len(df_jan.columns))

df_jan["duration"] = df_jan["tpep_dropoff_datetime"] - df_jan["tpep_pickup_datetime"]
df_jan.duration = df_jan.duration.apply(lambda td: td.total_seconds() / 60)

# get standard deviation of df_jan
std_jan = df_jan.duration.std()
print(std_jan)

df_jan_droped = df_jan[(df_jan.duration >= 1) & (df_jan.duration <= 60)]
# fraction of the records left after you dropped
print(len(df_jan_droped) / len(df_jan) * 100)

df = df_jan_droped

categorical = ["PULocationID", "DOLocationID"]

df[categorical] = df[categorical].astype(str)

train_dicts = df[categorical].to_dict(orient="records")

dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)
# Number of columns of X train matrix
X_train.shape


target = "duration"
y_train = df[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)
# Calculate RMSE
print(mean_squared_error(y_train, y_pred, squared=False))

# Apply model to validation data
X_val = df_feb
X_val["duration"] = X_val["tpep_dropoff_datetime"] - X_val["tpep_pickup_datetime"]
X_val["duration"] = X_val.duration.apply(lambda td: td.total_seconds() / 60)
X_val = X_val[(X_val["duration"] >= 1) & (X_val["duration"] <= 60)]
y_val = X_val["duration"].values

keys = X_val.columns.tolist()
values = df_jan.columns.to_list()
combined = zip(keys, values)
dictionary = dict(combined)
X_val.rename(columns=dictionary, inplace=True)
X_val.columns
# Calculate dicts from validation data
X_val[categorical] = X_val[categorical].astype(str)
val_dicts = X_val[categorical].to_dict(orient="records")
# Calculate X val matrix
X_val = dv.transform(val_dicts)
X_val.shape
y_pred_val = lr.predict(X_val)
# Calculate RMSE of validation data
print(mean_squared_error(y_val, y_pred_val, squared=False))
