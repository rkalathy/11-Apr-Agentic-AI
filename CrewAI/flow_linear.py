import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.flow.flow import start, listen, Flow
from crewai_tools import SerpApiGoogleSearchTool, ScrapeWebsiteTool
from pydantic import BaseModel


class ContentState(BaseModel):
    topic: str = ""
    research_summary: str = ""
    analysis_summary: str = ""
    data_analysis: str = ""
    brief_summary: str = ""

load_dotenv()
# 0 = concise and deterministic responses, 0.5 = balanced creativity and coherence, 1 = highly creative and diverse responses
llm = LLM(model="openai/gpt-4o-mini", temperature=0) 
llm2 = LLM(model="openai/gpt-4o-mini", temperature=0.5)
llm3 = LLM(model="openai/gpt-4o-mini", temperature=1)

search = SerpApiGoogleSearchTool()
scrape = ScrapeWebsiteTool()

class ContentFlow(Flow[ContentState]):

    @start()
    def set_researcher_topic(self):
        # self.state.topic is already populated from kickoff(inputs={"topic": ...})
        print(f"Starting flow for topic: {self.state.topic}")

    @listen(set_researcher_topic)
    def researcher(self):
        researcher_agent = Agent(
            tools=[search, scrape],
            role="Web Researcher",
            goal=f"Conduct web research to gather the top 5 trends on: {self.state.topic}",
            backstory=(
                "You are a diligent and resourceful web researcher with a knack for "
                "finding accurate and relevant information online. You synthesize "
                "information into clear and concise summaries."
            ),
            llm=llm,
            verbose=True,
        )
        task = Task(
            agent=researcher_agent,
            description=(
                f"Conduct web research on the top 5 trends for: {self.state.topic}. "
                "Include market trends and consumer preferences."
            ),
            expected_output=(
                f"A comprehensive summary of the top 5 trends for {self.state.topic}, "
                "including market trends and consumer preferences."
            ),
        )
        crew = Crew(agents=[researcher_agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.research_summary = result.raw

    @listen(researcher)
    def analyst(self):
        analyst_agent = Agent(
            role="Industry Analyst",
            goal=f"Analyze research and provide insights on: {self.state.topic}",
            backstory=(
                "You are an experienced industry analyst with a deep understanding "
                "of market trends and dynamics."
            ),
            llm=llm,
            verbose=True,
        )
        task = Task(
            agent=analyst_agent,
            description=(
                f"Analyze the following research and provide insights on the top 5 trends "
                f"for: {self.state.topic}.\n\nResearch:\n{self.state.research_summary}"
            ),
            expected_output=(
                f"A detailed analysis of the top 5 trends for {self.state.topic}, "
                "including market trends and consumer preferences."
            ),
        )
        crew = Crew(agents=[analyst_agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.analysis_summary = result.raw

    @listen(analyst)
    def writer(self):
        writer_agent = Agent(
            role="Brief Writer",
            goal=f"Write a concise brief on the top 5 trends for: {self.state.topic}",
            backstory=(
                "You are a skilled writer with a talent for distilling complex "
                "information into clear and concise summaries for a general audience."
            ),
            llm=llm,
            verbose=True,
        )
        task = Task(
            agent=writer_agent,
            description=(
                f"Write a brief summary of the top 5 trends for: {self.state.topic}.\n\n"
                f"Research:\n{self.state.research_summary}\n\n"
                f"Analysis:\n{self.state.analysis_summary}"
            ),
            expected_output=(
                f"A clear and concise summary of the top 5 trends for {self.state.topic}, "
                "suitable for a general audience."
            ),
        )
        crew = Crew(agents=[writer_agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.brief_summary = result.raw


flow = ContentFlow()
result = flow.kickoff(inputs={"topic": "iphone in 2026 and return tabular data"})
print(result.brief_summary)