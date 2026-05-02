import os
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.team import Team
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

web_agent = Agent(
    model=Gemini(id="gemini-2.5-pro"),
    tools=[DuckDuckGoTools()],
    instructions="You are a helpful assistant that answers questions using the web search on duck duck go.",
    markdown=True,
    db=SqliteDb("memory.db"),
    add_history_to_context=True,
    num_history_runs=5,
    debug_mode=True,
    role="You are responsible web search"
)

finance_agent = Agent(
    model=Gemini(id="gemini-2.5-pro"),
    tools=[YFinanceTools()],
    instructions="You are a helpful assistant that answers questions about the stock market.",
    markdown=True,
    db=SqliteDb("memory.db"),
    add_history_to_context=True,
    num_history_runs=5,
    debug_mode=True,
    role="You are responsible for answering questions about the stock market"
)

team = Team( 
    name="Finance and Web Team",
    model=Gemini(id="gemini-2.5-pro"),
    instructions="You are a team of agents that work together to answer questions. The web agent is responsible for answering questions using the web search on duck duck go. The finance agent is responsible for answering questions about the stock market. You should work together to answer the user's question.",
    debug_mode=True,
    members=[finance_agent, web_agent] )

response = team.run("What is the current price of AAPL and what is the latest news about it?")

print(response.content)