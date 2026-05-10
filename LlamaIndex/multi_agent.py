"""Multi-agent demo: Researcher and Writer hand off via AgentWorkflow."""

import asyncio
from dotenv import load_dotenv
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

load_dotenv(".env")

llm = OpenAI(model="gpt-4o-mini")


def record_research_notes(topic: str, notes: str) -> str:
    """Save the researcher's findings so the writer can use them."""
    return f"Notes saved for '{topic}':\n{notes}"


def submit_final_report(report: str) -> str:
    """Submit the writer's final report. Use this when the report is ready."""
    return f"Report submitted:\n\n{report}"


researcher = FunctionAgent(
    name="researcher",
    description="Researches a topic and records concise notes for the writer.",
    system_prompt=(
        "You are a researcher. Given a topic, produce 4-6 bullet points of "
        "factual notes covering key history, current state, and future outlook. "
        "Save them with record_research_notes, then hand off to writer."
    ),
    tools=[FunctionTool.from_defaults(fn=record_research_notes)],
    llm=llm,
    can_handoff_to=["writer"],
)

writer = FunctionAgent(
    name="writer",
    description="Turns research notes into a polished short report.",
    system_prompt=(
        "You are a technical writer. Read the researcher's notes and write a "
        "tight 3-paragraph report. Submit it with submit_final_report."
    ),
    tools=[FunctionTool.from_defaults(fn=submit_final_report)],
    llm=llm,
    can_handoff_to=["researcher"],
)

workflow = AgentWorkflow(
    agents=[researcher, writer],
    root_agent="researcher",
)


async def main():
    topic = "the rise of small language models in 2025-2026"
    print(f">>> Topic: {topic}\n")
    response = await workflow.run(user_msg=f"Write a short report on: {topic}")
    print("\n=== Final Output ===")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
