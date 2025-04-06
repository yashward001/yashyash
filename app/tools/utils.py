from datetime import datetime

from openbb import obb
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolNode

import pandas as pd
import json


def wrap_dataframe(df: pd.DataFrame) -> str:
    """
    Wrap a DataFrame with a special marker for UI rendering.
    
    This allows the UI to parse and render the DataFrame as a table.
    
    Args:
        df (pd.DataFrame): DataFrame to be wrapped
    
    Returns:
        str: Wrapped DataFrame representation
    """
    try:
        # Convert DataFrame to a JSON string for easy parsing
        df_json = df.to_json()
        return f"\n<observation>\n[TABLE:{df_json}]\n</observation>\n"
    except Exception as e:
        return f"\n<observation>Error converting DataFrame: {str(e)}\n</observation>\n"


def fetch_stock_data(
    symbol: str, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    return obb.equity.price.historical(
        symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        provider="yfinance",
    ).to_df()


def fetch_sp500_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    return obb.equity.price.historical(
        "^GSPC",
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        provider="yfinance",
    ).to_df()


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )