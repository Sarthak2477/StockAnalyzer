
import http.client
import json
import matplotlib.pyplot as plt
from uagents import Agent, Context
import pandas as pd
import numpy as np
import json as json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from uagents import Agent, Context
from messages.message import Message

Analyzer = Agent(
    name="analyzer",
    seed="analyzer agent for the stock prediction system"
)

# Process fetched data and make predictions

def analyze_stock_data(data):
    print("analyzing stock data")
    time_series_data = data['Time Series (Daily)']
    df = pd.DataFrame(time_series_data).transpose()
    df = df.apply(pd.to_numeric)

    # Use 'close' price for prediction
    data = df[['4. close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare data for LSTM
    sequence_length = 10
    sequences = []
    next_prices = []

    for i in range(len(scaled_data) - sequence_length):
        sequences.append(scaled_data[i:i + sequence_length])
        next_prices.append(scaled_data[i + sequence_length])

    X = np.array(sequences)
    y = np.array(next_prices)

    # Split data into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)

    # Make predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Create a simple Matplotlib line plot
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[split + sequence_length:], df['4. close'].values[split + sequence_length:], label='Actual Prices')
    plt.plot(df.index[split + sequence_length:], predicted_prices, label='Predicted Prices')
    plt.title('Actual vs. Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Calculate Mean Squared Error
    mse = np.mean(np.square(predicted_prices.flatten() - df['4. close'].values[split + sequence_length:]))
    print("Mean Squared Error:", mse)

    # Generate investment recommendations based on predicted prices
    if predicted_prices[-1] > predicted_prices[-2]:
        recommendation = "Buy Signal"
    elif predicted_prices[-1] < predicted_prices[-2]:
        recommendation = "Sell Signal"
    else:
        recommendation = "Hold Signal"

    print("Investment Recommendation:", recommendation)

def load_data_from_json(filename):
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{filename}': {e}")
        return None
    
@Analyzer.on_message(model=Message)
async def message_handler(ctx: Context, sender: str, msg: Message):
    ctx.logger.info("Analyzing data...")
    try:
        ctx.logger.info("data found in the storage.") 
        analyze_stock_data(load_data_from_json('fetched_data.json'))  # No need for data.decode("utf-8")
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)