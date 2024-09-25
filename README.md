Stock Analysis and Prediction Tool
Overview
This tool analyzes and predicts stock prices using historical data, technical indicators, and sentiment analysis. It generates visualizations to help investors make informed decisions by understanding potential future stock movements and correlations with economic indicators.




Prerequisites
Before running the tool, please ensure you have the following:

A Computer with Internet Access

The tool fetches data from the internet, so a stable connection is necessary.
Python 3 Installed

Python is the programming language used for this tool.
You can download Python from the official website: Download Python.
During installation, make sure to check the box that says "Add Python to PATH".




Setup Instructions
1. Download the Project Files
Option 1: Clone the Repository

If you're familiar with Git, you can clone the repository using the command:

bash
Copy code
git clone https://github.com/yourusername/yourrepository.git
Option 2: Download as ZIP

Visit the repository page.
Click on the "Code" button and select "Download ZIP".
Extract the downloaded ZIP file to a folder on your computer.


2. Install Python 3
If you haven't installed Python 3:

Download it from here https://www.python.org/downloads/.
Run the installer and follow the on-screen instructions.
Ensure you select the option to "Add Python to PATH" during installation.



3. Install Required Libraries
Open the Command Prompt (Windows) or Terminal (Mac/Linux).

Navigate to the directory where you extracted the project files:- cd path/to/your/project/folder



Install the required Python libraries by running:


pip install pandas yfinance seaborn matplotlib fredapi plotly prophet feedparser nltk


Note: If you encounter any errors during installation, you might need to upgrade pip by running: python -m pip install --upgrade pip



4. Set Up NLTK Data
Run the following command to download necessary NLTK data:

python -m nltk.downloader vader_lexicon



5. Prepare the Ticker List Files
Ensure that the following CSV files are placed in the same directory as the script:

Russel_3000_ticker_list.csv
TSX List.csv
TSXV.csv
CSE Listings.csv
India stock list.csv
These files contain lists of valid stock tickers and are required for the script to verify the tickers you input.



6. Run the Script
In the Command Prompt or Terminal, ensure you're in the project directory.

Run the script by typing:

python stock_predictor.py



7. Enter Tickers When Prompted
The script will prompt you to:


Enter the tickers (comma-separated):


Type in the stock ticker symbols you want to analyze, separated by commas.

For example:

Enter the tickers (comma-separated): AAPL

or

Enter the tickers (comma-separated): AAPL, MSFT
etc


8. View the Outputs
The script will perform the analysis and generate visualizations, which include:

Stock Analysis Graphs:

Displays Bollinger Bands, Prophet Forecasts, and Bullish/Bearish Signals.
Sentiment Analysis Graphs:

Shows the contribution of specific positive and negative terms to the overall sentiment.
Correlation Heatmaps:

Illustrates the correlation between stock prices and various economic indicators.
Note: The script may take some time to run, especially if analyzing multiple tickers.



***Understanding the Visualizations***
Stock Analysis Graphs
Bollinger Bands:

Blue lines representing the upper and lower bands.
Helps identify volatility and potential price movements.
Prophet Forecast:

Purple line showing predicted future stock prices.
Bullish Signals:

Green triangles indicating potential upward trends.
Bearish Signals:

Red triangles indicating potential downward trends.
Sentiment Analysis Graphs
Positive Terms (Blue Bars):

Words that contribute positively to investor sentiment.
Negative Terms (Red Bars):

Words that contribute negatively to investor sentiment.
Understanding the Graph:

The height of each bar represents the average sentiment contribution of that term.
Helps identify key factors influencing market perception.



--------------------------------------------------------------------------------


When the output is generated , first the snentiment graph will appear, close that and then the stock prediction graph adn economic indicator grid will appear.
























