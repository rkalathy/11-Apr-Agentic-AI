import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

load_dotenv()
# 0 = concise and deterministic responses, 0.5 = balanced creativity and coherence, 1 = highly creative and diverse responses
llm = LLM(model="openai/gpt-4o-mini", temperature=0) 
llm2 = LLM(model="openai/gpt-4o-mini", temperature=0.5)
llm3 = LLM(model="openai/gpt-4o-mini", temperature=1)

search = SerperDevTool()
scrape = ScrapeWebsiteTool()

researcher = Agent(
    tools=[search, scrape],
    role="Web Researcher",
    goal=("Conduct web research to gather information " 
        "top 5 on a given topic. the is going - {topic}"),
    backstory=("You are a diligent and resourceful web researcher"
    " with a knack for finding accurate and relevant "
    "information online. You have experience in using various"
    " search engines, databases, and online resources to gather"
    " data efficiently. Your research skills are complemented by"
    " your ability to critically evaluate sources and synthesize "
    "information into clear and concise summaries."),
    llm=llm,
    verbose=True,
)

analyst = Agent(
    role="Industry Analyst",
    goal=("Analyze the gathered information and provide insights on the top 5"
    " trends in the industry. The topic is - {topic}"),
    backstory=("You are an experienced industry analyst "
    "with a deep understanding of market trends and"
    " dynamics. "),
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Brief Writer",
    goal=("Write a brief summary of the top 5"
    " trends in the industry. The topic is - {topic}"),
    backstory=("You are a skilled writer with a talent for"
    " distilling complex information into clear and concise"
    " summaries. You have experience in creating engaging"
    " content for various audiences."),
    llm=llm,
    verbose=True,
)


researcher_task = Task(
    agent=researcher,
    name="Research on {topic}",
    description="Conduct web research to gather information on the top 5 trends in the smartphone industry, with a focus on the {topic}. The research should include information on market trends, consumer preferences",
    expected_output="A comprehensive summary of the top 5 trends in the smartphone industry, with a focus on the {topic}, including market trends and consumer preferences."
)

analyst_task = Task(
    context=[researcher_task],
    agent=analyst,
    name="Analyze {topic} Trends",
    description="Analyze the gathered information and provide insights on the top 5 trends in the smartphone industry, with a focus on the {topic}. The analysis should include market trends and consumer preferences",
    expected_output="A detailed analysis of the top 5 trends in the smartphone industry, with a focus on the {topic}, including market trends and consumer preferences."
)

writer_task = Task(
    context=[researcher_task, analyst_task],
    agent=writer,
    name="Write Brief on {topic} Trends",
    description="Write a brief summary of the top 5 trends in the smartphone industry, with a focus on the {topic}. The summary should be clear and concise, suitable for a general audience.",
    expected_output="A clear and concise summary of the top 5 trends in the smartphone industry, with a focus on the {topic}, suitable for a general audience.",
    output_file="trends_summary.md"
)

crew = Crew(
    name="Story Writers",
    agents=[researcher, analyst, writer], 
    verbose=True, 
    tasks=[researcher_task, analyst_task, writer_task],
    process=Process.sequential
)

result = crew.kickoff(inputs={"topic": "iphone in 2026"})
print(result.raw)