from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain.agents import tool

from app.features.chart import get_chart_base64
from app.tools.types import StockStatsInput


@tool(args_schema=StockStatsInput)
def get_stock_chart_analysis(symbol: str) -> str:
    """Using the chart data, generate a technical analysis summary with visualizations."""

    try:
        chart_data = get_chart_base64(symbol)
        
        # Include multiple visualization representations
        visualization_markers = ""
        
        # Image representation
        if 'chart' in chart_data:
            visualization_markers += f"[CHART:{chart_data['chart']}]"
        
        # Plotly representation
        if 'plotly' in chart_data:
            visualization_markers += f"[PLOTLY:{repr(chart_data['plotly'])}]"
        
        llm = ChatAnthropic(model_name="claude-3-opus-20240229", max_tokens=4096)
        analysis = llm.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Analyze the following stock chart image and provide a technical analysis summary:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{chart_data['chart']}"
                            },
                        },
                    ]
                )
            ]
        )
        
        return f"\n<observation>\n{visualization_markers}{analysis}\n</observation>\n"
    except Exception as e:
        return f"\n<observation>\nError: {e}\n</observation>\n"