"""Single-agent demo: an Agent with two simple tools."""

import asyncio
from datetime import datetime
from typing import Annotated
from dotenv import load_dotenv
from pydantic import Field
from agent_framework.openai import OpenAIChatClient

load_dotenv(".env")


def calculate_tip(
    bill_amount: Annotated[float, Field(description="The bill amount in USD")],
    tip_percent: Annotated[float, Field(description="The tip percentage, 0-100")],
) -> str:
    """Calculate tip and total for a restaurant bill."""
    tip = bill_amount * (tip_percent / 100)
    total = bill_amount + tip
    return f"Tip: ${tip:.2f}, Total: ${total:.2f}"


def current_time() -> str:
    """Return the current date and time as an ISO string."""
    return datetime.now().isoformat(timespec="seconds")


async def main():
    agent = OpenAIChatClient(model="gpt-4o-mini").as_agent(
        name="basic_agent",
        instructions=(
            "You are a helpful assistant. Use the available tools to answer the user. "
            "Always state what tool you used and its raw result."
        ),
        tools=[calculate_tip, current_time],
    )

    questions = [
        "What's the time right now?",
        "If my bill is $87.50 and I want to leave an 18% tip, what do I pay?",
    ]
    for q in questions:
        print(f"\n>>> {q}")
        response = await agent.run(q)
        print(response.text)


if __name__ == "__main__":
    asyncio.run(main())
