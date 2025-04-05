from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional, Dict, Any

from langchain.agents import tool
import yfinance as yf
from openbb import obb
import quantstats as qs
import pandas as pd

from app.features.technical import add_technicals
from app.features.screener import fetch_custom_universe
from app.tools.utils import wrap_dataframe
from app.tools.types import StockStatsInput


@lru_cache(maxsize=128)
def fetch_stock_data(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch stock data using yfinance directly with caching to improve performance.
    
    Args:
        symbol (str): The stock symbol to fetch data for.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str, optional): The end date in 'YYYY-MM-DD' format. Defaults to today.
    
    Returns:
        pd.DataFrame: DataFrame with standardized column names.
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        
        if df.empty:
            # Try alternative source via OpenBB if yfinance returns empty data
            try:
                df = obb.equity.price.historical(symbol=symbol, start_date=start_date).to_df()
            except Exception:
                return pd.DataFrame()
        
        # Standardize column names to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Make sure 'close' is available
        if 'adj close' in df.columns:
            df["close"] = df["adj close"]
        
        df.index = pd.to_datetime(df.index)
        return df
    
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


@tool(args_schema=StockStatsInput)
def get_stock_price_history(symbol: str) -> str:
    """Fetch a Stock's Price History by Symbol."""
    try:
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        df = fetch_stock_data(symbol, start_date)

        if df.empty:
            return "\n<observation>\nNo data found for the given symbol\n</observation>\n"

        df = add_technicals(df)
        df = df[-30:][::-1]  # Last 30 days, reversed

        return wrap_dataframe(df)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_quantstats(symbol: str) -> str:
    """Fetch a Stock's Portfolio Analytics For Quants by Symbol."""
    try:
        start_date = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        df = fetch_stock_data(symbol, start_date)

        if df.empty:
            return f"\n<observation>\nNo data found for the given symbol {symbol}\n</observation>\n"

        # Using cached data instead of downloading again via quantstats
        stock_returns = df['close'].pct_change().dropna()
        
        # Still need benchmark data
        bench_ticker = yf.Ticker("^GSPC")
        bench_df = bench_ticker.history(start=start_date)
        bench_returns = bench_df['Close'].pct_change().dropna()
        
        # Convert both indexes to UTC to make them comparable
        stock_returns.index = stock_returns.index.tz_localize(None)
        bench_returns.index = bench_returns.index.tz_localize(None)
        
        # Align the dates
        common_dates = stock_returns.index.intersection(bench_returns.index)
        stock_returns = stock_returns.loc[common_dates]
        bench_returns = bench_returns.loc[common_dates]
        
        stats = qs.reports.metrics(
            stock_returns, mode="full", benchmark=bench_returns, display=False
        )

        return f"\n<observation>\n{stats}\n</observation>\n"
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"

@tool
def get_gainers() -> str:
    """Fetch Top Price Gainers in the Stock Market."""
    try:
        gainers = obb.equity.discovery.gainers(sort="desc").to_df()

        if gainers.empty:
            return "\n<observation>\nNo gainers found\n</observation>\n"

        return wrap_dataframe(gainers)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool
def get_losers() -> str:
    """Fetch Stock Market's Top Losers."""
    try:
        losers = obb.equity.discovery.losers(sort="desc").to_df()

        if losers.empty:
            return "\n<observation>\nNo losers found\n</observation>\n"

        return wrap_dataframe(losers)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_ratios(symbol: str) -> str:
    """Fetch an Extensive Set of Financial and Accounting Ratios for a Given Company Over Time."""
    try:
        # First try with OpenBB
        try:
            trades = obb.equity.fundamental.ratios(symbol=symbol).to_df()
            if not trades.empty:
                return wrap_dataframe(trades)
        except Exception:
            pass
            
        # Fallback to yfinance
        ticker = yf.Ticker(symbol)
        financials = ticker.financials
        
        if financials.empty:
            return f"\n<observation>\nNo ratio data found for the given symbol {symbol}\n</observation>\n"
        
        # Process yfinance data to create ratios
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        
        # Basic ratios calculation
        ratios = {}
        if not income_stmt.empty and not balance_sheet.empty:
            try:
                # Example ratio calculations - expand as needed
                if 'Net Income' in income_stmt.index and 'Total Assets' in balance_sheet.index:
                    ratios['ROA'] = income_stmt.loc['Net Income'] / balance_sheet.loc['Total Assets']
                
                if 'Net Income' in income_stmt.index and 'Total Stockholder Equity' in balance_sheet.index:
                    ratios['ROE'] = income_stmt.loc['Net Income'] / balance_sheet.loc['Total Stockholder Equity']
                
                if 'Total Revenue' in income_stmt.index and 'Net Income' in income_stmt.index:
                    ratios['Net Margin'] = income_stmt.loc['Net Income'] / income_stmt.loc['Total Revenue']
                
                ratios_df = pd.DataFrame(ratios)
                return wrap_dataframe(ratios_df)
            except Exception:
                pass
                
        return f"\n<observation>\nUnable to calculate ratios for {symbol}\n</observation>\n"
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_key_metrics(symbol: str) -> str:
    """Fetch Fundamental Metrics by Symbol."""
    try:
        # Try OpenBB first
        try:
            metrics = obb.equity.fundamental.metrics(
                symbol=symbol, with_ttm=True, provider="yfinance"
            ).to_df()
            
            if not metrics.empty:
                return wrap_dataframe(metrics[::-1])
        except Exception:
            pass
            
        # Fallback to direct yfinance
        ticker = yf.Ticker(symbol)
        
        # Compile key metrics from various yfinance data
        info = ticker.info
        metrics_dict: Dict[str, Any] = {}
        
        # Extract key metrics from info dictionary
        key_fields = [
            'marketCap', 'beta', 'trailingPE', 'forwardPE', 'dividendYield',
            'returnOnAssets', 'returnOnEquity', 'revenueGrowth', 
            'grossMargins', 'operatingMargins', 'profitMargins',
            'debtToEquity', 'currentRatio', 'quickRatio'
        ]
        
        for field in key_fields:
            if field in info:
                metrics_dict[field] = info[field]
        
        # Create DataFrame
        metrics_df = pd.DataFrame([metrics_dict])
        
        if metrics_df.empty:
            return f"\n<observation>\nNo metrics data found for {symbol}\n</observation>\n"
            
        return wrap_dataframe(metrics_df)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_sector_info(symbol: str) -> str:
    """Fetch a Company's General Information By Symbol. This includes company name, industry, and sector data."""
    try:
        # Try OpenBB first
        try:
            profile = obb.equity.profile(symbol=symbol).to_df()
            if not profile.empty:
                return wrap_dataframe(profile)
        except Exception:
            pass
            
        # Fallback to direct yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract relevant information
        profile_dict = {
            'Name': info.get('shortName', ''),
            'Sector': info.get('sector', ''),
            'Industry': info.get('industry', ''),
            'Country': info.get('country', ''),
            'Exchange': info.get('exchange', ''),
            'Website': info.get('website', ''),
            'Business Summary': info.get('longBusinessSummary', '')
        }
        
        profile_df = pd.DataFrame([profile_dict])
        
        if profile_df.empty:
            return f"\n<observation>\nNo profile data found for {symbol}\n</observation>\n"
            
        return wrap_dataframe(profile_df)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_valuation_multiples(symbol: str) -> str:
    """Fetch a Company's Valuation Multiples by Symbol."""
    try:
        # Try OpenBB first
        try:
            df = obb.equity.fundamental.multiples(symbol=symbol).to_df()
            if not df.empty:
                return wrap_dataframe(df)
        except Exception:
            pass
            
        # Fallback to direct yfinance
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract relevant multiples
        multiples_dict = {
            'P/E': info.get('trailingPE', None),
            'Forward P/E': info.get('forwardPE', None),
            'PEG Ratio': info.get('pegRatio', None),
            'Price/Sales': info.get('priceToSalesTrailing12Months', None),
            'Price/Book': info.get('priceToBook', None),
            'Enterprise Value/Revenue': info.get('enterpriseToRevenue', None),
            'Enterprise Value/EBITDA': info.get('enterpriseToEbitda', None)
        }
        
        # Remove None values
        multiples_dict = {k: v for k, v in multiples_dict.items() if v is not None}
        
        multiples_df = pd.DataFrame([multiples_dict])
        
        if multiples_df.empty:
            return f"\n<observation>\nNo valuation multiples found for {symbol}\n</observation>\n"
            
        return wrap_dataframe(multiples_df)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool
def get_stock_universe() -> str:
    """Fetch Bullish Trending Stocks Universe from FinViz."""
    try:
        return wrap_dataframe(fetch_custom_universe())
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_financials(symbol: str) -> str:
    """Fetch a Company's Financial Statements by Symbol."""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get income statement, balance sheet, and cash flow
        income_stmt = ticker.income_stmt
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        
        if income_stmt.empty and balance_sheet.empty and cashflow.empty:
            return f"\n<observation>\nNo financial data found for {symbol}\n</observation>\n"
        
        result = "\n<observation>\n"
        
        if not income_stmt.empty:
            result += "Income Statement:\n"
            result += str(income_stmt) + "\n\n"
            
        if not balance_sheet.empty:
            result += "Balance Sheet:\n"
            result += str(balance_sheet) + "\n\n"
            
        if not cashflow.empty:
            result += "Cash Flow Statement:\n"
            result += str(cashflow) + "\n"
            
        result += "</observation>\n"
        
        return result
    except Exception as e:
        return f"\n<observation>\nError fetching financials for {symbol}: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_stock_recommendations(symbol: str) -> str:
    """Fetch Analyst Recommendations for a Company by Symbol."""
    try:
        ticker = yf.Ticker(symbol)
        recommendations = ticker.recommendations
        
        if recommendations is None or recommendations.empty:
            return f"\n<observation>\nNo analyst recommendations found for {symbol}\n</observation>\n"
        
        # Sort by date, most recent first
        recommendations = recommendations.sort_index(ascending=False)
        
        return wrap_dataframe(recommendations)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"


@tool(args_schema=StockStatsInput)
def get_institutional_holders(symbol: str) -> str:
    """Fetch Institutional Holders of a Company's Stock by Symbol."""
    try:
        ticker = yf.Ticker(symbol)
        holders = ticker.institutional_holders
        
        if holders is None or holders.empty:
            return f"\n<observation>\nNo institutional holders data found for {symbol}\n</observation>\n"
        
        return wrap_dataframe(holders)
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"