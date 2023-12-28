# Stock Prediction System with Data Fetcher and Analyzer

This Python program is designed to predict stock prices using historical data fetched from the Alpha Vantage API and analyzed with a Long Short-Term Memory (LSTM) model. It consists of two main components: DataFetcher and Analyzer.

# Components:
  DataFetcher:
    Responsible for fetching stock data from the Alpha Vantage API at regular intervals.
    Asks for user input (stock symbol and time series) if not provided.
    Stores fetched data as a JSON file (fetched_data.json).
    Utilizes an agent called data fetcher.
  Analyzer:
    Analyzes fetched stock data to predict future stock prices.
    Utilizes LSTM model for predictions.
    Trains the model on fetched data and generates investment recommendations based on predicted prices.
    Implements an agent called analyzer for processing and analyzing the data.
  Usage:
    Run the DataFetcher module, which fetches stock data and stores it in fetched_data.json.
    Run the Analyzer module to analyze the fetched data, make predictions, and generate investment recommendations.
# How to Run:
  1. Clone the repository
  2. Start the Virtual Environment
      `Scripts\activate`
  3. Install dependencies
  4. Run main.py
      `python main.py`
# Notes:
  Adjust the model parameters and training epochs in the analyze_stock_data function for better predictions.
  Ensure fetched_data.json exists in the same directory before running the Analyzer.
# Dependencies:
  tensorflow, pandas, numpy, sklearn for data processing and LSTM modeling.
  uagents library for agent-based interaction.
