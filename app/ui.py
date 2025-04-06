from typing import Callable, TypeVar

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig
from openbb import obb
from dotenv import load_dotenv

import os
os.environ["ANTHROPIC_API_KEY"] = "."

from chains.clear_results import with_clear_container
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.chains.agent import create_anthropic_agent_graph

import io
import base64
import uuid
import warnings
import inspect
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

warnings.filterwarnings("ignore")

load_dotenv()

obb.account.login(pat=os.environ.get("OPENBB_TOKEN"), remember_me=True)
obb.user.credentials.tiingo_token = os.environ.get("TIINGO_API_KEY")
obb.user.credentials.fmp_api_key = os.environ.get("FMP_API_KEY")
obb.user.credentials.intrinio_api_key = os.environ.get("INTRINIO_API_KEY")
obb.user.credentials.fred_api_key = os.environ.get("FRED_API_KEY")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

st.set_page_config(
    page_title="Financial Chat | The Trading Dude Abides",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

T = TypeVar("T")


def get_streamlit_cb(parent_container: DeltaGenerator):
    def decor(fn: Callable[..., T]) -> Callable[..., T]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> T:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = StreamlitCallbackHandler(parent_container=parent_container)

    for name, fn in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if name.startswith("on_"):
            setattr(st_cb, name, decor(fn))

    return st_cb


def render_visualization(content: str, answer_container):
    """
    Parse the response and render appropriate visualizations.
    
    Markers:
    - [CHART: base64_encoded_image]
    - [TABLE: pandas_dataframe]
    - [PLOTLY: plotly_json_config]
    """
    # Check for chart markers
    if "[CHART:" in content and "]" in content:
        # Extract base64 encoded image
        start = content.index("[CHART:") + 7
        end = content.index("]", start)
        base64_image = content[start:end]
        
        try:
            # Decode and display image
            image_bytes = base64.b64decode(base64_image)
            answer_container.image(image_bytes, use_column_width=True)
            
            # Remove the chart marker from the content
            content = content.replace(f"[CHART:{base64_image}]", "")
        except Exception as e:
            st.error(f"Error rendering chart: {e}")
    
    # Check for Plotly chart markers
    if "[PLOTLY:" in content and "]" in content:
        try:
            start = content.index("[PLOTLY:") + 8
            end = content.index("]", start)
            plotly_config = content[start:end]
            
            # Parse the Plotly configuration
            fig = go.Figure(eval(plotly_config))
            answer_container.plotly_chart(fig, use_container_width=True)
            
            # Remove the Plotly marker from the content
            content = content.replace(f"[PLOTLY:{plotly_config}]", "")
        except Exception as e:
            st.error(f"Error rendering Plotly chart: {e}")
    
    # Check for table markers
    if "[TABLE:" in content and "]" in content:
        try:
            start = content.index("[TABLE:") + 7
            end = content.index("]", start)
            table_str = content[start:end]
            
            # Convert table string back to DataFrame
            df = pd.read_json(table_str)
            answer_container.dataframe(df, use_container_width=True)
            
            # Remove the table marker from the content
            content = content.replace(f"[TABLE:{table_str}]", "")
        except Exception as e:
            st.error(f"Error rendering table: {e}")
    
    # Write the remaining content
    if content.strip():
        answer_container.write(content)


if "graph" not in st.session_state:
    st.session_state.graph = create_anthropic_agent_graph()

st.title("Financial Chat, your AI financial advisor 📈")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

with st.form(key="form"):
    user_input = st.text_input(
        "Please ask your question here:",
        placeholder="Please give me a full analysis of Apple",
    )
    submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()

    output_container.markdown(f"**User:** {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})

    answer_container = output_container.chat_message("assistant", avatar="💸")
    st_callback = get_streamlit_cb(answer_container)

    cfg = RunnableConfig(recursion_limit=150)
    cfg["callbacks"] = [st_callback]
    cfg["configurable"] = {"thread_id": uuid.uuid4()}

    question = {"messages": ("user", user_input)}

    response = st.session_state.graph.invoke(question, cfg)
    answer = response["messages"][-1].content

    # Use the new render_visualization function
    render_visualization(answer, answer_container)

    st.session_state.messages.append({"role": "assistant", "content": answer})