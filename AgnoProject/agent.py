import os
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb

chat = Agent(
    model=Gemini(id="gemini-2.5-pro"),
    instructions="You are a helpful assistant that answers questions about the stock market.",
    markdown=True,
    db=SqliteDb("memory.db"),
    add_history_to_context=True,
    num_history_runs=5,
)

session_id      = "praveen-session"
session_id_2    = "amit-session"

print("Starting Agents...")
response_praveen_1 = chat.run("What is the current price of AAPL?", session_id=session_id)
response_praveen_2 = chat.run("What is the current price of MSFT?", session_id=session_id)

response_amit_1 = chat.run("What is the current price of Google?", session_id=session_id_2)
response_amit_2 = chat.run("What is the current price of Oracle?", session_id=session_id_2)

print("Praveen Session:")
print(response_praveen_1)
print(response_praveen_2)   
print("\nAmit Session:")
print(response_amit_1)
print(response_amit_2)

response_praveen_3 = chat.run("what stocks did i ask about?", session_id=session_id)
response_amit_3 = chat.run("what stocks did i ask about?", session_id=session_id_2)

print("\nPraveen Session:")
print(response_praveen_3)
print("\nAmit Session:") 
print(response_amit_3)      