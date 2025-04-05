from typing import List, Any, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict , Field
from langchain_core.messages import HumanMessage, AIMessage
from langserve import add_routes
from openbb import obb
from dotenv import load_dotenv

import os
import warnings
import pandas as pd

from app.chains.agent import create_anthropic_agent_graph

warnings.filterwarnings("ignore")

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = "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"
obb.account.login(pat=os.environ.get("OPENBB_TOKEN"), remember_me=True)
obb.user.credentials.tiingo_token = os.environ.get("TIINGO_API_KEY")
obb.user.credentials.fmp_api_key = os.environ.get("FMP_API_KEY")
obb.user.credentials.intrinio_api_key = os.environ.get("INTRINIO_API_KEY")
obb.user.credentials.fred_api_key = os.environ.get("FRED_API_KEY")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

app = FastAPI(
    title="Financial Chat",
    version="1.0",
    description="The Trading Dude Abides",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

graph = create_anthropic_agent_graph()


class AgentInput(BaseModel):
    messages: List[Union[HumanMessage, AIMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
        extra={"widget": {"type": "chat", "input": "messages"}},
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentOutput(BaseModel):
    output: Any


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.get("/health")
async def health():
    return {"status": "ok"}


add_routes(
    app,
    graph,
    path="/chat",
    input_type=AgentInput,
    output_type=AgentOutput,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
