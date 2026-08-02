from dotenv import load_dotenv
from typing import List

load_dotenv(override=True)
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="Thr agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


@tool
def search_tool(query: str) -> str:
    """Search the web for information about the query"""
    print(f"Search results for {query} ")
    return tavily.search(query=query)

tavily = TavilySearch()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# tools = [search_tool]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course! Search Agent!")
    query = "What is the capital of France?"
    query = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(result)


if __name__ == "__main__":
    main()
