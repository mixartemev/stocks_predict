# Load Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
company = 'AAPL'
start = dt.datetime(2021, 1, 15)
end = dt.datetime(2021, 2, 10)
train_prices_col = web.DataReader(company, 'yahoo', start, end)['Close'].values.reshape(-1, 1)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_col = scaler.fit_transform(train_prices_col)
prediction_days = 5
x_train_matrix = []
y_train_row = []

for x in range(prediction_days, len(train_data_col)):
    x_train_matrix.append(train_data_col[x - prediction_days:x, 0])  # input
    y_train_row.append(train_data_col[x, 0])  # output

x_train_matrix, y_train_row = np.array(x_train_matrix), np.array(y_train_row)
x_train_matrix = x_train_matrix.reshape((x_train_matrix.shape[0], x_train_matrix.shape[1], 1))

# Build Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_matrix.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of next close price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train_matrix, y_train_row, epochs=5, batch_size=32)

# Test the model accuracy on existed data
test_start = dt.datetime(2021, 2, 11)
test_end = dt.datetime.now()

actual_prices_col = web.DataReader(company, 'yahoo', test_start, test_end)['Close'].values.reshape(-1, 1)

model_inputs = np.vstack((train_prices_col[-prediction_days:], actual_prices_col))
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days: x, 0])

x_test = np.array(x_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

predicted_data_col = model.predict(x_test)
predicted_prices_col = scaler.inverse_transform(predicted_data_col)

# Plot predicted and actual prices
plt.plot(actual_prices_col, color='black', label=f'Actual {company} price')
plt.plot(predicted_prices_col, color='green', label=f'Predicted {company} price')
plt.title(f'{company} Share Price')
plt.xlabel('Time')
plt.xlabel('Price')
plt.legend()
plt.show()

# Predict tomorrow
real_data = [model_inputs[len(model_inputs) - prediction_days: len(model_inputs), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Tomorrow {company} price: {prediction[0][0]}")
