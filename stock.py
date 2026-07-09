import yfinance as yf
import pandas as pd


class StockAnalyzer:
    def get_income_statement(self):
        return self.stock.financials

    def get_balance_sheet(self):
        return self.stock.balance_sheet

    def get_cashflow(self):
        return self.stock.cashflow


    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)

    def get_info(self):
        info = self.stock.info

        data = {
            "Company": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Current Price": info.get("currentPrice", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "PE Ratio": info.get("trailingPE", "N/A"),
            "EPS": info.get("trailingEps", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Website": info.get("website", "N/A"),
            "Summary": info.get("longBusinessSummary", "N/A"),
        }

        return data

    def get_history(self, period="1y"):
        history = self.stock.history(period=period)

        if history.empty:
            return history

        history["SMA20"] = history["Close"].rolling(20).mean()
        history["SMA50"] = history["Close"].rolling(50).mean()
        history["EMA20"] = history["Close"].ewm(span=20).mean()

        return history