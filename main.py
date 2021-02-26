# Load Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
from os.path import exists
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Init vars
company = 'TSLA'
start_train = dt.date(2015, 1, 1)
end = dt.date.today()
pred_days = 20
test_days = 100

# Load Data
prices_col = web.DataReader(company, 'yahoo', start_train, end)['Close'].values.reshape(-1, 1)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_col = scaler.fit_transform(prices_col)

x_matrix = []
y_row = []
for i in range(pred_days, len(data_col)):
    x_matrix.append(data_col[i-pred_days:i, 0])  # input
    y_row.append(data_col[i, 0])  # output

x_matrix, y_row = np.array(x_matrix), np.array(y_row)
x_matrix = x_matrix.reshape((x_matrix.shape[0], x_matrix.shape[1], 1))

# Build Model
if exists('mdl-20.h5'):
    model = load_model('mdl-20.h5')
else:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(pred_days, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of next close price
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_matrix[:-test_days], y_row[:-test_days], epochs=25, batch_size=32)
    model.save('mdl.h5')

# Make predictions on test data
predicted_data_col = model.predict(x_matrix[-test_days:])
predicted_prices_col = scaler.inverse_transform(predicted_data_col)
actual_prices = prices_col[-test_days:]
dif = predicted_prices_col - actual_prices

# Predict tomorrow
real_data = np.array([y_row[-pred_days:]])
real_data = real_data.reshape((1, pred_days, 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)[0][0]
print(f"Tomorrow {company} price: {prediction}")

# Plot predicted and actual prices
plt.plot(prices_col[-test_days:], color='black', label=f'Actual {company} price')
plt.plot(predicted_prices_col, color='green', label=f'Predicted {company} price')
dif_color = np.where(dif < 0, 'b', 'r')
plt.bar(range(dif.size), dif[:, 0], label='Loss')
plt.title(f'Tomorrow {company} Price: ${prediction:.2f}')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
