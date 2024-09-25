import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from fredapi import Fred
import plotly.graph_objects as go
from prophet import Prophet  # Make sure Prophet is installed
import plotly.io as pio
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os  # For file handling
import logging  # For logging errors

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class StockAnalysis:
    def __init__(self, tickers, start_date, end_date):
        """
        Initialize the StockAnalysis object.

        Parameters:
            tickers (list): List of tickers for which to perform the analysis.
            start_date (str): Start date for historical data (YYYY-MM-DD).
            end_date (str): End date for historical data (YYYY-MM-DD).
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.all_traces = []
        self.closing_prices_df = pd.DataFrame()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def fetch_stock_data(self, ticker):
        """
        Fetch historical stock data for the given ticker from Yahoo Finance.

        Parameters:
            ticker (str): Stock ticker symbol.

        Returns:
            pandas.DataFrame: DataFrame containing the historical stock data.
        """
        return yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)

    def prepare_prophet_data(self, stock_data):
        prophet_data = stock_data.reset_index()[["Date", "Close"]]
        prophet_data.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
        return prophet_data

    def calculate_bollinger_bands(self, stock_data):
        window = 20
        std_dev = 2
        stock_data["MA"] = stock_data["Close"].rolling(window=window).mean()
        stock_data["Upper"] = stock_data["MA"] + std_dev * stock_data["Close"].rolling(window=window).std()
        stock_data["Lower"] = stock_data["MA"] - std_dev * stock_data["Close"].rolling(window=window).std()
        return stock_data

    def identify_bullish_bearish_signals(self, stock_data):
        bullish_signals = stock_data[stock_data["Close"] > stock_data["Upper"]]["Close"]
        bearish_signals = stock_data[stock_data["Close"] < stock_data["Lower"]]["Close"]
        return bullish_signals, bearish_signals

    def create_traces_for_ticker(self, ticker, stock_data, forecast, bullish_signals, bearish_signals):
        traces = [
            go.Scatter(x=stock_data.index, y=stock_data["Upper"], mode="lines", name=f"{ticker} Upper Bollinger Band", line=dict(color="blue")),
            go.Scatter(x=stock_data.index, y=stock_data["Lower"], mode="lines", name=f"{ticker} Lower Bollinger Band", line=dict(color="blue")),
            go.Scatter(x=stock_data.index, y=stock_data["Close"], mode="lines", name=f"{ticker} Closing Price", line=dict(color="black")),
            go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name=f"{ticker} Prophet Forecast", line=dict(color="purple")),
            go.Scatter(x=bullish_signals.index, y=bullish_signals, mode="markers", name=f"{ticker} Bullish Signal", marker=dict(symbol="triangle-up", size=10, color="green")),
            go.Scatter(x=bearish_signals.index, y=bearish_signals, mode="markers", name=f"{ticker} Bearish Signal", marker=dict(symbol="triangle-down", size=10, color="red")),
            go.Scatter(x=forecast["ds"], y=forecast["trend"], mode="lines", name=f"{ticker} Trend Line", line=dict(color="orange", dash="dash"))
        ]
        return traces

    def visualize_sentiment_graph(self, term_contributions):
        terms = list(term_contributions.keys())
        contributions = [term_contributions[term] for term in terms]
        colors = ['blue' if contribution >= 0 else 'red' for contribution in contributions]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(terms, contributions, color=colors)
        plt.xlabel('Terms')
        plt.ylabel('Contribution to Sentiment')
        plt.title('Sentiment Graph')

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=4, label='Positive Sentiment'),
            Line2D([0], [0], color='red', lw=4, label='Negative Sentiment')
        ]
        plt.legend(handles=legend_elements)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def analyze_sentiment_terms(self, ticker):
        """
        Analyze the contribution of specific terms to overall sentiment.

        Parameters:
            ticker (str): Stock ticker symbol.

        Returns:
            dict: Dictionary with terms as keys and their sentiment contributions as values.
        """
        # Define the positive and negative terms
        negative_terms = ['bad', 'marginally', 'gross', 'killing', 'unlikely', 'hard', 'issue', 'sharply', 'slowest', 'stress']
        positive_terms = ['eased', 'good', 'optimistic', 'popular', 'recovery', 'top']
        all_terms = negative_terms + positive_terms

        rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'
        feed = feedparser.parse(rss_url)
        term_contributions = {term: 0 for term in all_terms}
        total_occurrences = {term: 0 for term in all_terms}

        for entry in feed.entries:
            text = (entry.title + ' ' + entry.summary).lower()
            for term in all_terms:
                if term in text:
                    sentiment_score = self.sentiment_analyzer.polarity_scores(term)['compound']
                    term_contributions[term] += sentiment_score
                    total_occurrences[term] += 1

        # Average the contributions
        for term in term_contributions:
            if total_occurrences[term] > 0:
                term_contributions[term] /= total_occurrences[term]
            else:
                term_contributions[term] = 0

        return term_contributions

    def visualize_all_ticker_data(self):
        for ticker in self.tickers:
            try:
                stock_data = self.fetch_stock_data(ticker)
                if stock_data.empty:
                    logging.warning(f"No stock data available for {ticker}. Skipping.")
                    continue
                self.closing_prices_df[ticker] = stock_data["Close"]
                prophet_data = self.prepare_prophet_data(stock_data)
                model = Prophet(daily_seasonality=True)
                model.fit(prophet_data)
                future = model.make_future_dataframe(periods=365)  # 365 days forecast
                forecast = model.predict(future)
                stock_data = self.calculate_bollinger_bands(stock_data)
                bullish_signals, bearish_signals = self.identify_bullish_bearish_signals(stock_data)
                traces = self.create_traces_for_ticker(ticker, stock_data, forecast, bullish_signals, bearish_signals)
                self.all_traces.extend(traces)

                # Analyze sentiment terms and visualize sentiment graph
                term_contributions = self.analyze_sentiment_terms(ticker)
                self.visualize_sentiment_graph(term_contributions)
            except Exception as e:
                logging.error(f"Error processing data for {ticker}: {e}")
                continue

        if self.all_traces:
            fig = go.Figure(data=self.all_traces)
            today_date = datetime.today().strftime('%Y-%m-%d')
            fig.add_shape(
                type="line",
                x0=today_date,
                x1=today_date,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="gray", width=1.5, dash="dash"),
                name="Today's Date"
            )
            fig.add_annotation(
                x=today_date,
                y=0.9,
                xref="x",
                yref="paper",
                text="Today's Date",
                showarrow=True,
                font=dict(size=12, color="black"),
                align="center",
                arrowhead=1,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black",
                ax=0,
                ay=-40
            )
            fig.update_layout(
                title=f"({', '.join(self.tickers)}) - Bollinger Bands, Prophet Forecast, and Bullish/Bearish Signals",
                xaxis_title="Date",
                template="plotly_white"
            )
            fig.show()
        else:
            logging.info("No data to display.")

class CorrelationVisualizer:
    def __init__(self, tickers, economic_indicators, start_date, today):
        """
        Initialize the CorrelationVisualizer object.

        Parameters:
            tickers (list): List of tickers for which you want to calculate the correlation matrix.
            economic_indicators (dict): A dictionary containing economic indicators and their series IDs.
            start_date (str): Start date in the format 'YYYY-MM-DD'.
            today (str): End date in the format 'YYYY-MM-DD'.
        """
        self.tickers = tickers
        self.economic_indicators = economic_indicators
        self.start_date = start_date
        self.today = today

    def fetch_economic_data(self):
        """
        Fetch economic indicator data from FRED API.

        Returns:
            pandas.DataFrame: DataFrame containing economic indicator data.
        """
        API_KEY = "630bf87ab1df091dc6417dd3e918e101"  # Replace with your FRED API key
        fred_client = Fred(api_key=API_KEY)

        economic_data = {}
        for series_id, series_name in self.economic_indicators.items():
            try:
                data = fred_client.get_series(series_id, start_date=self.start_date, end_date=self.today)
                economic_data[series_name] = data
            except Exception as e:
                logging.error(f"Error fetching data for {series_name}: {e}")

        economic_df = pd.DataFrame(economic_data)
        economic_df.index = pd.to_datetime(economic_df.index)
        return economic_df

    def fetch_stock_data(self, ticker):
        """
        Fetch historical stock price data for the chosen ticker using yfinance.

        Parameters:
            ticker (str): The ticker symbol for which to fetch historical stock price data.

        Returns:
            pandas.DataFrame: DataFrame containing historical stock price data for the chosen ticker.
        """
        stock_data = yf.download(ticker, start=self.start_date, end=self.today)
        return stock_data

    def calculate_correlation_matrix(self, economic_df, stock_data, ticker):
        """
        Calculate the correlation matrix for the chosen ticker.

        Parameters:
            economic_df (pandas.DataFrame): DataFrame containing economic indicator data.
            stock_data (pandas.DataFrame): DataFrame containing historical stock price data for the chosen ticker.
            ticker (str): The ticker symbol for which to calculate the correlation matrix.

        Returns:
            pandas.DataFrame: Correlation matrix for the chosen ticker.
        """
        stock_data = stock_data.copy()
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data = stock_data.resample('D').ffill()

        merged_df = pd.merge(stock_data[['Close']], economic_df, left_index=True, right_index=True, how='inner')
        merged_df.rename(columns={'Close': ticker + ' Close Price'}, inplace=True)

        correlation_matrix = merged_df.corr()
        return correlation_matrix

    def visualize_correlation_heatmap(self, ticker, correlation_matrix):
        """
        Visualize the correlation matrix as a heatmap using Seaborn.

        Parameters:
            ticker (str): The ticker symbol for which to visualize the correlation matrix.
            correlation_matrix (pandas.DataFrame): Correlation matrix for the chosen ticker.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=True, yticklabels=True)
        plt.title(f"Correlation Matrix for {ticker} and Economic Indicators")
        plt.xlabel("Variables")
        plt.ylabel("Variables")
        plt.show()

    def calculate_all_correlation_matrices(self):
        """
        Calculate correlation matrices for all tickers.

        Returns:
            dict: A dictionary containing correlation matrices for each ticker.
        """
        economic_df = self.fetch_economic_data()
        correlation_matrices = {}  # Dictionary to store correlation matrices for each ticker
        for ticker in self.tickers:
            try:
                stock_data = self.fetch_stock_data(ticker)
                if stock_data.empty:
                    logging.warning(f"No stock data available for {ticker}. Skipping.")
                    continue
                correlation_matrix = self.calculate_correlation_matrix(economic_df, stock_data, ticker)
                correlation_matrices[ticker] = correlation_matrix
            except Exception as e:
                logging.error(f"Error processing data for {ticker}: {e}")
                continue
        return correlation_matrices

    def visualize_all_correlations(self):
        """
        Fetches data and visualizes correlation for all tickers.
        """
        correlation_matrices = self.calculate_all_correlation_matrices()
        for ticker, correlation_matrix in correlation_matrices.items():
            self.visualize_correlation_heatmap(ticker, correlation_matrix)

def read_tickers_from_files():
    """
    Read ticker symbols from specified CSV files in the current directory.

    Returns:
        list: A combined list of ticker symbols from all available files.
    """
    filenames = [
        'Russel_3000_ticker_list.csv',
        'TSX List.csv',
        'TSXV.csv',
        'CSE Listings.csv',
        'India stock list.csv'
    ]

    tickers = []
    for filename in filenames:
        if os.path.isfile(filename):
            try:
                df = pd.read_csv(filename)
                # Assuming the ticker symbols are in a column named 'Ticker' or similar
                ticker_column = None
                for col in df.columns:
                    if 'ticker' in col.lower() or 'symbol' in col.lower():
                        ticker_column = col
                        break
                if ticker_column:
                    file_tickers = df[ticker_column].astype(str).str.strip().tolist()
                    tickers.extend(file_tickers)
                    logging.info(f"Loaded {len(file_tickers)} tickers from {filename}")
                else:
                    logging.warning(f"No ticker column found in {filename}. Skipping.")
            except Exception as e:
                logging.error(f"Error reading {filename}: {e}")
        else:
            logging.warning(f"File {filename} not found in the current directory.")
    return tickers

if __name__ == "__main__":
    # Common Economic Indicators from FRED with their series IDs
    economic_indicators = {
        'HOUST': "Housing Starts",
        'UNRATE': "Unemployment Rate",
        'CPIAUCSL': "Consumer Price Index for All Urban Consumers: All Items",
        'GDPC1': "Real Gross Domestic Product (Quarterly)",
        'FEDFUNDS': "Effective Federal Funds Rate",
        'PCE': "Personal Consumption Expenditures",
        'INDPRO': "Industrial Production Index",
        # Add more economic indicators and their series IDs as needed
    }

    # Read tickers from files
    available_tickers = read_tickers_from_files()
    if not available_tickers:
        logging.error("No tickers loaded from files. Please ensure the CSV files are in the current directory.")
        exit()
    available_tickers_set = set(available_tickers)

    # Prompt user for list of tickers
    tickers_input = input("Enter the tickers (comma-separated): ")
    input_tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

    # Verify that the input tickers exist in the available tickers
    tickers = []
    for ticker in input_tickers:
        if ticker in available_tickers_set:
            tickers.append(ticker)
        else:
            logging.warning(f"Ticker {ticker} not found in the available ticker lists.")

    if not tickers:
        logging.error("No valid tickers provided. Exiting.")
        exit()

    start_date = (datetime.today() - timedelta(days=4 * 365)).strftime('%Y-%m-%d')
    today = datetime.today().strftime('%Y-%m-%d')

    # Create StockAnalysis object
    stock_analysis = StockAnalysis(tickers, start_date, end_date=today)

    # Perform analysis and visualize all ticker data
    stock_analysis.visualize_all_ticker_data()

    # Create CorrelationVisualizer object and visualize correlations for all tickers
    visualizer = CorrelationVisualizer(tickers, economic_indicators, start_date, today)
    visualizer.visualize_all_correlations()
