import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib;
from holoviews.plotting.bokeh.styles import alpha

matplotlib.use('Qt5Agg')
import seaborn as sns

df = pd.read_csv('data/time_series_data.csv')

def data_understanding(dataframe, num=5, plot=False):
    print('########## Dataset Shape ##########')
    print('Number of rows:', dataframe.shape[0], '\nNumber of columns:', dataframe.shape[1])
    print('\n')
    print(f'########## First {num} Rows ##########')
    print(dataframe.head(num))
    print('\n')
    print('########## Column Names ##########')
    print(dataframe.columns)
    print('\n')
    print('########## Unique & Null values ##########')
    data = pd.DataFrame(index=dataframe.columns)
    data["Unique_values"] = dataframe.nunique()
    data["Null_values"] = dataframe.isnull().sum()
    print(data)
    print('\n')
    print('########## Variable Types ##########')
    print(dataframe.info())
    print('\n')
    print('########## Summary Statistics ##########')
    print(dataframe.describe().transpose())
    if plot:
        for col in dataframe.columns:
            if dataframe[col].dtype != "O":
                sns.histplot(x=col, data=dataframe, kde=True, bins=100)
                plt.show(block=True)
    if plot:
        for col in dataframe.columns:
            if dataframe[col].nunique() < 60:
                sns.countplot(x=col, data=dataframe)
                plt.xticks(rotation=90)
                plt.show(block=True)

data_understanding(df, num=5, plot=False)


df["Datetime"] = pd.to_datetime(df["Datetime"],format='%d-%m-%Y %H:%M')

df["Datetime"].min()
# Timestamp('2012-08-25 00:00:00')
df["Datetime"].max()
# Timestamp('2014-09-25 02:00:00')
# 2 yil 1 ay 1 gun 2 saat'lik bir zaman aralığı var.
df.isnull().sum()

# ID          0
# Datetime    0
# Count       0
# dtype: int64

# Veri setine genel bakis
df['ID'].nunique()
# 18288 adet farkli ID var. ID degiskeni esktra bir bilgi vermiyor.
df.drop('ID', axis=1, inplace=True)
# ID degiskenini sildik.
df.set_index('Datetime', inplace=True)
# Datetime degiskenini  zaman serisi analizi yapacagimiz icin index olarak atadik.

# Zaman serisinde count degiskeninin zamanla nasil degistigine bakalim.
fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(x=df.index, y='Count', data=df, ax=ax, label='Count')
ax.set_title('Count of Sensor')
ax.set_xlabel('Datetime')
ax.set_ylabel('Count')
ax.grid(True)
ax.legend()
fig.savefig('images/count_time_series.png')
plt.show()

# 'Count' degiskeninin istatistiklerine bakalim.
df['Count'].describe()
# count    18288.000000
# mean       138.958115
# std        153.467461
# min          2.000000
# 25%         22.000000
# 50%         84.000000
# 75%        210.000000
# max       1244.000000
# Name: Count, dtype: float64

# 'Count' degiskeninin dağılımına bakalim.
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Count'], kde=True, bins=100)
ax.set_title('Count Distribution')
ax.set_xlabel('Count')
ax.set_ylabel('Frequency')
fig.savefig('images/count_distribution.png')
plt.show()

# Count degiskeni saga carpik bir dagilim gosteriyor.
# Bu durumda log donusumu yapabiliriz.
# Log donusumu yapmanin amaci, verinin normal dagilima
# daha yakin olmasini saglamaktir.

df['Log_Count'] = np.log1p(df['Count'])

# Log donusumu yapip yeni bir degisken olusturduk. Histograma tekrar bakalim.

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['Log_Count'], kde=True, bins=40)
ax.set_title('Logarithmic Count Distribution')
ax.set_xlabel('Log_Count')
ax.set_ylabel('Frequency')
fig.savefig('images/log_count_distribution.png')
plt.show()

# Log donusumu sonrasi degiskenin normal dagilima daha yakin oldugunu goruyoruz.
# Bu durumda log donusumu yaparak devam edecegiz.

from sklearn.ensemble import IsolationForest

# Isolation Forest algoritmasi kullanarak aykiri degerleri tespit edecegiz.
clf = IsolationForest(contamination=0.05, random_state=42)

X = df['Count'].values.reshape(-1, 1)

clf.fit(X)
df['Anomaly'] = clf.predict(X)

df['Anomaly'].value_counts()
# Anomaly
#  1    17375
# -1      913
# Name: count, dtype: int64

df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})
df['Anomaly'].value_counts()
# Anomaly
# 0    17375
# 1      913
# Name: count, dtype: int64


#Degerleri [0,1] arasina olcekleyerek tekrar deneyelim.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

clf_scaled = IsolationForest(contamination=0.05, random_state=42)

clf_scaled.fit(X_scaled)
df['Anomaly_scaled'] = clf_scaled.predict(X_scaled)

df['Anomaly_scaled'].value_counts()

# Anomaly_scaled
#  1    17375
# -1      913
# Name: count, dtype: int64
# Sayilarda bir degisim olmadi.

# Isolation forest kullanarak aykiri degerleri tespit ettik. Aykiri degerleri gorselle
# tirerek inceleyelim.

fig, ax = plt.subplots(figsize=(15, 6))
sns.scatterplot(x=df[df['Anomaly'] == 1].index, y='Count',
                data=df[df['Anomaly'] == 1], color='red', label='Anomaly')
sns.lineplot(x=df.index, y='Count', data=df, label='Count', alpha=0.7)

ax.set_title('Isolation Forest Anomaly Count')
ax.set_xlabel('Datetime')
ax.set_ylabel('Count')
ax.grid(True)
ax.legend()
fig.savefig('images/anomaly_detection.png')
plt.show()

# Bir de Z-Score metodu ile aykiri degerleri tespit edelim.
df['zscore'] = (df['Count'] - df['Count'].mean()) / df['Count'].std()
df.head()

# zscore degiskeninin histogramina bakalim.
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['zscore'], kde=True, bins=40)
ax.set_title('Z-Score Distribution')
ax.set_xlabel('Z-Score')
ax.set_ylabel('Frequency')
fig.savefig('images/zscore_distribution.png')
plt.show()

# Grafige bakarak threshold degerini 3 olarak belirleyelim.
threshold = 3
df['Anomaly_zscore'] = np.where(df['zscore'] > threshold, 1, 0)
df['Anomaly_zscore'].value_counts()

# Anomaly_zscore
# 0    17975
# 1      313
# Name: count, dtype: int64
# z-score metodu ile aykiri degerleri tespit ettik. Isolation Forest algoritmasina gore daha
# az aykiri deger tespit etti.
# z-score kullanarak bulunan aykiri degerleri gorsellesitirelim.


fig, ax = plt.subplots(figsize=(15, 6))
sns.scatterplot(x=df[df['Anomaly_zscore'] == 1].index, y='Count',
                data=df[df['Anomaly_zscore'] == 1], color='red', label='Anomaly_zscore')
sns.lineplot(x=df.index, y='Count', data=df, label='Count', alpha=0.7)
ax.set_title('z-score Anomaly Count')
ax.set_xlabel('Datetime')
ax.set_ylabel('Count')
ax.grid(True)
ax.legend()
fig.savefig('images/anomaly_detection_zscore.png')
plt.show()

# Mevsimsellik-trend ayirimini anlamak icin veriyi bilesenlere bolmek gerekir. Bu islem icin
# STL decomposition yontemini kullanacagiz.
from statsmodels.tsa.seasonal import STL

stl = STL(df['Count'])
result = stl.fit()

fig, ax = plt.subplots(4,1, figsize=(15, 6))


ax[0].plot(df['Count'], label='Original')
ax[0].legend(loc='upper left')

ax[1].plot(result.trend, label='Trend')
ax[1].legend(loc='upper left')

ax[2].plot(result.seasonal, label='Seasonality')
ax[2].legend(loc='upper left')

ax[3].plot(result.resid, label='Residuals')
ax[3].legend(loc='upper left')


plt.tight_layout()
fig.savefig('images/stl_decomposition.png')
plt.show()

# STL decomposition sonucunda mevsimsellik ve trend bilesenlerini ayirdik.
# Datanin duragan olup olmadigini anlamak icin ADF hipotez testi yapacagiz.

from statsmodels.tsa.stattools import adfuller
sonuc = adfuller(df['Count'])

print('ADF Statistic: %f' % sonuc[0])
print('p-value: %f' % sonuc[1])
# ADF Statistic: -4.456561
# p-value: 0.000235

# Hipotez testinin sonucuna gore p-value degeri 0.000235 oldugu icin
# H0 hipotezi reddedilir ve serinin duragan oldugu sonucuna variriz.

# Zaman serisinin dinamik ozelliklerini yakalamak, gecmis verilerin etkisini analiz etmek
# ve tahmin gucumuzu artirmak adina gecikmeli degerleri ekleyecegiz.

df['Lag_1'] = df['Count'].shift(1)
df['Lag_2'] = df['Count'].shift(2)
df['Lag_3'] = df['Count'].shift(3)
df.head()
#                      Count  Log_Count  Anomaly  ...  Lag_1  Lag_2  Lag_3
# Datetime                                        ...
# 2012-08-25 00:00:00      8   2.197225        0  ...    NaN    NaN    NaN
# 2012-08-25 01:00:00      2   1.098612        0  ...    8.0    NaN    NaN
# 2012-08-25 02:00:00      6   1.945910        0  ...    2.0    8.0    NaN
# 2012-08-25 03:00:00      2   1.098612        0  ...    6.0    2.0    8.0
# 2012-08-25 04:00:00      2   1.098612        0  ...    2.0    6.0    2.0
df = df.dropna() # Gecikmeli degerler ekledikten sonra eksik degerleri sildik
df.columns


df_processed = df.drop(['Log_Count', 'Anomaly', 'Anomaly_scaled','zscore'], axis = 1)


# Datayi train ve test olarak ayiriyoruz. Zaman serisi oldugu icin train_test_split kullanmiyoruz.
# Veriyi kronolojik olarak ayiriyoruz. Bunun icin datanin uzunluguna gore bir bolme yapacagiz.

split = int(len(df)*0.8)
train = df_processed.iloc[:split]
test = df_processed.iloc[split:]

train.head()
test.head()

# Train ve test verilerini ayirdik. Simdi modelimizi olusturabiliriz.

from sklearn.linear_model import LinearRegression

# ozellikler cikartalim. Dataframe'den gun ay ve yil bilgilerini cikartalim.

train['hour'] = train.index.hour
train['day_of_week'] = train.index.dayofweek
train['day_of_year'] = train.index.dayofyear

test['hour'] = test.index.hour
test['day_of_week'] = test.index.dayofweek
test['day_of_year'] = test.index.dayofyear

# Hedef degiskeni belirleyelim.

train_target = train['Count']
test_target = test['Count']

train_data = train.drop(['Count','Lag_1', 'Lag_2', 'Lag_3'], axis=1)
test_data = test.drop(['Count','Lag_1', 'Lag_2', 'Lag_3'], axis=1)

train_data.head()

# Modeli olusturalim.
lr = LinearRegression()
lr.fit(train_data, train_target)

preds_lr = lr.predict(test_data)

from sklearn.metrics import mean_absolute_error, mean_squared_error

print(f"Mean Absolute Error (MAE): {mean_absolute_error(test_target, preds_lr)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(test_target, preds_lr)}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(test_target, preds_lr))}")

# Mean Absolute Error (MAE): 218.30237561971657
# Mean Squared Error (MSE): 62746.37630525256
# Root Mean Squared Error (RMSE): 250.49226795502602

#Modelin tahminlerini gorsellestirelim.
fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(x=train.index, y=train_target, label='Training Data')
sns.lineplot(x=test.index, y=test_target, label='Actual')
sns.lineplot(x=test.index, y=preds_lr, label='Predicted')
ax.set_title('Linear Regression - Actual vs Predicted')
ax.set_xlabel('Datetime')
ax.set_ylabel('Count')
ax.grid(True)
ax.legend()
fig.savefig('images/linear_reg_actual_vs_predicted.png')
plt.show()

# Modelin tahminleri kotu cikti.
# Baska bir model deneyelim.

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(train_data, train_target)

preds_rf = rf.predict(test_data)

rf.fit(train_data, train_target)

preds_rf = rf.predict(test_data)


print(f"Mean Absolute Error (MAE): {mean_absolute_error(test_target, preds_rf)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(test_target, preds_rf)}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(test_target, preds_rf))}")

# Gorsellestirelim.

fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(x=train.index, y=train_target, label='Training Data')
sns.lineplot(x=test.index, y=test_target, label='Actual')
sns.lineplot(x=test.index, y=preds_rf, label='Predicted')
ax.set_title('Random Forest Regressor - Actual vs Predicted')
ax.set_xlabel('Datetime')
ax.set_ylabel('Count')
ax.grid(True)
ax.legend()
fig.savefig('images/random_forest_actual_vs_predicted.png')
plt.show()

# Random Forest modeli de basarili sonuc vermedi.
# XGBoost modelini deneyelim.
# XGBoost modeli deneyelim.


from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(train_data, train_target)

preds_xgb = xgb.predict(test_data)

xgb.fit(train_data, train_target)

preds_xgb = xgb.predict(test_data)

print(f"Mean Absolute Error (MAE): {mean_absolute_error(test_target, preds_xgb)}")
print(f"Mean Squared Error (MSE): {mean_squared_error(test_target, preds_xgb)}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(test_target, preds_xgb))}")

fig, ax = plt.subplots(figsize=(15, 6))
sns.lineplot(x=train.index, y=train_target, label='Training Data')
sns.lineplot(x=test.index, y=test_target, label='Actual')
sns.lineplot(x=test.index, y=preds_xgb, label='Predicted')
ax.set_title('XGBRegressor - Actual vs Predicted')
ax.set_xlabel('Datetime')
ax.set_ylabel('Count')
ax.grid(True)
ax.legend()
fig.savefig('images/xgboost_actual_vs_predicted.png')
plt.show()


# Klasik machine learning modelleri iyi tahminler vermedi.
# LSTM Modeliyle neural network deneyelim.


from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df_processed[['Count']])

# Veriyi LSTM icin uygun hale getirmeliyiz. n adim onceki zaman adimini kullanarak bir sonraki
# adimi tahmin etmek istiyoruz. Bunu icin bir diziler butunu yaratmaliyiz.

def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

sequence_length = 24  # 24 saatlik bir aralik sectik
X, y = create_sequences(scaled_df, sequence_length)

scaled_df.shape
#(18285, 1)

X.shape
# (18261, 24, 1)

# Oncekine benzer train-test ayrimi yapalim
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]


# LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=32))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

epochs = 20
batch_size = 32

# Train
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

# Prediction ve predict edilen degerleri tersine scale donusumu ile istenilen forma getirme

predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Medel degerlendirmesi

mae = mean_absolute_error(y_test_inv, predicted)
mse = mean_squared_error(y_test_inv, predicted)
rmse = np.sqrt(mean_squared_error(y_test_inv, predicted))
print(f"LSTM MAE: {mae}")
print(f"LSTM MSE: {mse}")
print(f"LSTM RMSE: {rmse}")

# Modelin loss grafigini cizdirelim

loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.grid(True)
plt.savefig('images/loss_curve.png')
plt.show()


# Plot results
plt.figure(figsize=(15,6))
plt.plot(df_processed.index, df_processed['Count'].values, label='Train data')
plt.plot(df_processed.index[-len(predicted):], y_test_inv, label='Actual')
plt.plot(df_processed.index[-len(predicted):], predicted, label='Predicted', alpha=0.5)
plt.title('LSTM - Actual vs Predicted')
plt.xlabel('Datetime')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig('images/lstm_actual_vs_predicted.png')
plt.show()

