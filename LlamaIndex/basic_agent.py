"""Single-agent demo: a FunctionAgent with two simple tools."""

import asyncio
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

load_dotenv(".env")


def calculate_tip(bill_amount: float, tip_percent: float) -> str:
    """Calculate tip and total for a restaurant bill."""
    tip = bill_amount * (tip_percent / 100)
    total = bill_amount + tip
    return f"Tip: ${tip:.2f}, Total: ${total:.2f}"


def current_time() -> str:
    """Return the current date and time as an ISO string."""
    return datetime.now().isoformat(timespec="seconds")


tools = [
    FunctionTool.from_defaults(fn=calculate_tip),
    FunctionTool.from_defaults(fn=current_time),
]

agent = FunctionAgent(
    tools=tools,
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt=(
        "You are a helpful assistant. Use the available tools to answer the user. "
        "Always state what tool you used and its raw result."
    ),
)


async def main():
    questions = [
        "What's the time right now?",
        "If my bill is $87.50 and I want to leave an 18% tip, what do I pay?",
    ]
    for q in questions:
        print(f"\n>>> {q}")
        response = await agent.run(q)
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
