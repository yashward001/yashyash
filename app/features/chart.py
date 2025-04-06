import io
import os
import base64
import tempfile
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import pyimgur
import seaborn as sns

from dotenv import load_dotenv
from openbb import obb

from app.features.technical import add_technicals

load_dotenv()

IMGUR_CLIENT_ID = os.environ.get("IMGUR_CLIENT_ID")
IMGUR_CLIENT_SECRET = os.environ.get("IMGUR_CLIENT_SECRET")

sns.set_style("whitegrid")


def create_plotly_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Generate a Plotly chart for stock data visualization.

    Args:
    - df (pd.DataFrame): Stock data with columns like 'open', 'high', 'low', 'close', 'SMA_50', 'SMA_200', 'RSI', 'ATR'
    - symbol (str): Stock symbol

    Returns:
    - go.Figure: Plotly figure for the stock
    """
    fig = sp.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=("", "", ""),
        row_heights=[0.6, 0.2, 0.2],
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            name="Price",
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
        ),
        row=1,
        col=1,
    )

    # SMA_50 trace
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["SMA_50"], mode="lines", name="50-day SMA", line=dict(color="blue")
        ),
        row=1,
        col=1,
    )
    # SMA_200 trace
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["SMA_200"], mode="lines", name="200-day SMA", line=dict(color="red")
        ),
        row=1,
        col=1,
    )

    # RSI trace
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["RSI"], mode="lines", name="RSI", line=dict(color="orange")
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # ATR trace
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["ATR"], mode="lines", name="ATR", line=dict(color="orange")
        ),
        row=3,
        col=1,
    )

    now = datetime.now().strftime("%m/%d/%Y")
    fig.update_layout(
        height=600,
        width=800,
        title_text=f"{symbol} | {now}",
        title_y=0.98,
        plot_bgcolor="lightgray",
        xaxis_rangebreaks=[
            dict(bounds=["sat", "mon"], pattern="day of week"),
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(size=10)
    )

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="ATR", row=3, col=1)

    return fig


def upload_image_to_imgur(buffer, symbol) -> str:
    """
    Uploads an image to Imgur.

    Args:
        buffer (io.BytesIO): The buffer containing the image data.
        symbol (str): The stock symbol associated with the image.

    Returns:
        str: The URL of the uploaded image on Imgur.
    """
    im = pyimgur.Imgur(
        IMGUR_CLIENT_ID, client_secret=IMGUR_CLIENT_SECRET, refresh_token=True
    )

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        tmp.write(buffer.getvalue())
        temp_path = tmp.name
        now = datetime.now().strftime("%m/%d/%Y")
        uploaded_image = im.upload_image(
            temp_path, title=f"{symbol} chart for {now}"
        )
        return uploaded_image.link


def plotly_fig_to_bytes(fig, filename="temp_plot.png"):
    """
    Convert a Plotly figure to a bytes object.

    Args:
        fig (plotly.graph_objs._figure.Figure): Plotly figure to convert
        filename (str): Temporary filename for saving the plot

    Returns:
        io.BytesIO: A bytes object containing the image data
    """
    fig.write_image(filename)
    with open(filename, "rb") as file:
        img_bytes = io.BytesIO(file.read())
    os.remove(filename)
    return img_bytes


def get_chart_base64(symbol: str) -> dict:
    """
    Generate a chart representation for a given stock symbol.
    Returns multiple representations to support different UI rendering methods.

    Args:
    symbol (str): The stock symbol to generate the chart for.

    Returns:
    dict: A dictionary containing chart representations.
    """
    try:
        start = datetime.now() - timedelta(days=365 * 2)
        start_date = start.strftime("%Y-%m-%d")
        df = obb.equity.price.historical(
            symbol, start_date=start_date, provider="yfinance"
        ).to_df()

        if df.empty:
            return {"error": "Stock data not found"}

        df = add_technicals(df)
        chart_data = create_plotly_chart(df, symbol)
        chart_bytes = plotly_fig_to_bytes(chart_data)
        chart_url = upload_image_to_imgur(chart_bytes, symbol)

        # Base64 encoded image for direct image rendering
        chart_base64 = base64.b64encode(chart_bytes.getvalue()).decode('utf-8')
        
        # Plotly figure configuration for interactive rendering
        plotly_config = chart_data.to_plotly_json()

        return {
            "chart": chart_base64,  # Image representation
            "url": chart_url,
            "plotly": plotly_config  # Interactive representation
        }
    except Exception as e:
        return {"error": f"Failed to generate chart: {str(e)}"}