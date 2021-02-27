# Load Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
from os.path import exists

from numpy import datetime64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Init vars
company = 'AAPL'
start_train = dt.date(2013, 1, 1)
end = dt.date.today()
pred_days = 50
test_days = 100
epochs = 50
lstm_units = 50
model_name = f'models/{company}/pd{pred_days}-e{epochs}-lu{lstm_units}.h5'

# Load Data
prices = web.DataReader(company, 'yahoo', start_train, end)['Close']
prices_col = prices.values.reshape(-1, 1)
nd: datetime64 = np.datetime64('2020', 'D')

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_col = scaler.fit_transform(prices_col)

x_matrix = []
y_row = []
for i in range(pred_days, len(data_col)):
    x_matrix.append(data_col[i - pred_days:i, 0])  # input
    y_row.append(data_col[i, 0])  # output

x_matrix, y_row = np.array(x_matrix), np.array(y_row)
x_matrix = x_matrix.reshape((x_matrix.shape[0], x_matrix.shape[1], 1))

# Build Model
if exists(model_name):
    model = load_model(model_name)
else:
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(pred_days, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of next close price
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_matrix[:-test_days], y_row[:-test_days], epochs=epochs, batch_size=32)
    model.save(model_name)

# Make predictions on test data
predicted_data_col = model.predict(x_matrix[-test_days:])
predicted_prices_col = scaler.inverse_transform(predicted_data_col)
actual_prices = prices_col[-test_days:]

loss = (predicted_prices_col - actual_prices)[:, 0]
lossSum = sum(abs(x) for x in loss) / loss.size

# Predict tomorrow
real_data = np.array([y_row[-pred_days:]])
real_data = real_data.reshape((1, pred_days, 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)[0][0]
print(f"Tomorrow {company} price: {prediction}")

# Plot predicted and actual prices
fig, ax1 = plt.subplots()
fig.suptitle(f'PredDays:{pred_days}, Epochs:{50}, StartYear:{2015}', fontsize=10)
plt.title(f'Tomorrow {company} Price: ${prediction:.2f}')
plt.xlabel('Days')
dates = [np.datetime64(x, 'D') for x in prices.index.values[-test_days:]]
dates.append(np.busday_offset(dates[-1], 1))
plt.xticks(ticks=range(0, loss.size+1, test_days//10), labels=dates[::test_days//10], fontsize=8, rotation=30)

ax1.bar(range(loss.size), loss, color=plt.get_cmap("coolwarm")(loss), label='Loss')
ax1.set_ylabel(f'Loss $ (Avg: {lossSum:.2f}, Last: {loss[-1]:.2f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(prices_col[-test_days:], color='black', label='Actual price')
ax2.plot(predicted_prices_col, color='green', label='Predicted price')
ax2.plot(loss.size+1, prediction, '*', color='green')
ax2.set_ylabel('Price $')
ax2.legend()

# fig.tight_layout()
plt.show()
