"""Multi-agent demo: Researcher chained to Writer via WorkflowBuilder."""

import asyncio
from dotenv import load_dotenv
from agent_framework import WorkflowBuilder, AgentExecutorResponse
from agent_framework.openai import OpenAIChatClient

load_dotenv(".env")


async def main():
    client = OpenAIChatClient(model="gpt-4o-mini")

    researcher = client.as_agent(
        name="researcher",
        instructions=(
            "You are a researcher. Given a topic, produce 4-6 concise bullet points "
            "covering key history, current state, and future outlook. "
            "Output ONLY the bullets — no preamble."
        ),
    )

    writer = client.as_agent(
        name="writer",
        instructions=(
            "You are a technical writer. You will receive research notes from the "
            "researcher. Turn them into a tight 3-paragraph report. "
            "Output ONLY the report — no preamble."
        ),
    )

    workflow = (
        WorkflowBuilder(start_executor=researcher)
        .add_edge(researcher, writer)
        .build()
    )

    topic = "the rise of small language models in 2025-2026"
    print(f">>> Topic: {topic}\n")

    final_text = None

    def handle(resp: AgentExecutorResponse):
        nonlocal final_text
        text = resp.agent_response.text
        print(f"\n--- {resp.executor_id} ---\n{text}")
        final_text = text

    async for event in workflow.run(f"Write a short report on: {topic}", stream=True):
        data = getattr(event, "data", None)
        if isinstance(data, AgentExecutorResponse):
            handle(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, AgentExecutorResponse):
                    handle(item)

    print("\n=== Final Output ===")
    print(final_text)


if __name__ == "__main__":
    asyncio.run(main())
