# python agent.py

import os
import json
from typing import Literal
from xml.etree.ElementTree import indent
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run an internet search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

llm = ChatOpenAI(model="gemini-2.0-flash", base_url="https://devs.ai/api/v1", api_key=os.environ["DEV_AI_KEY"])

agent = create_deep_agent(
    model = llm,
    tools=[internet_search],
    system_prompt=research_instructions,
)


# result = agent.invoke({"messages": [{"role": "user", "content": "Use internet_search to find information about first president of Kenya."}]})

# Print the agent's response
# print(result)
# print(result["messages"][-1].content)