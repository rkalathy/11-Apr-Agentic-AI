import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileReadTool

load_dotenv()

PRD_FILE_PATH = os.getenv("PRD_FILE_PATH", "prd.md")

llm = LLM(model="openai/gpt-4o-mini", temperature=0)

file_reader = FileReadTool()

# --- Agents ---

prd_parser = Agent(
    tools=[file_reader],
    role="PRD Parser",
    goal="Read and extract the full content of the PRD document at {prd_path}",
    backstory=(
        "You are a meticulous document analyst who reads product requirement "
        "documents and extracts their key sections: objectives, user stories, "
        "functional requirements, non-functional requirements, and success metrics."
    ),
    llm=llm,
    verbose=True,
)

tech_reviewer = Agent(
    role="Technical Feasibility Reviewer",
    goal="Evaluate the technical feasibility of the requirements in the PRD",
    backstory=(
        "You are a senior software architect with 15 years of experience shipping "
        "large-scale systems. You assess whether product requirements are technically "
        "sound, identify ambiguous or conflicting specs, flag under-specified "
        "integrations, and surface implementation risks before engineering starts."
    ),
    llm=llm,
    verbose=True,
)

ux_reviewer = Agent(
    role="UX and Product Reviewer",
    goal="Evaluate the user experience quality and product thinking in the PRD",
    backstory=(
        "You are a product designer and UX researcher who has shipped consumer and "
        "B2B products. You assess whether user stories are well-defined, whether the "
        "problem statement is clearly articulated, and whether the proposed solution "
        "genuinely addresses user needs. You flag missing edge cases and usability gaps."
    ),
    llm=llm,
    verbose=True,
)

business_reviewer = Agent(
    role="Business Strategy Reviewer",
    goal="Evaluate the business value, market fit, and strategic alignment of the PRD",
    backstory=(
        "You are a product strategist and former consultant who evaluates products "
        "through a business lens. You assess market opportunity, competitive "
        "differentiation, revenue impact, and whether success metrics are measurable "
        "and meaningful. You surface assumptions that need validation."
    ),
    llm=llm,
    verbose=True,
)

risk_assessor = Agent(
    role="Risk and Gaps Assessor",
    goal="Identify risks, missing sections, and open questions in the PRD",
    backstory=(
        "You are a technical program manager experienced in pre-mortem analysis. "
        "You look for what is missing or under-specified: security and privacy "
        "considerations, compliance requirements, dependency risks, rollout strategy, "
        "and anything that could derail delivery."
    ),
    llm=llm,
    verbose=True,
)

report_writer = Agent(
    role="Review Report Writer",
    goal="Synthesize all specialist feedback into a single structured review report",
    backstory=(
        "You are a clear, direct technical writer who produces concise executive "
        "summaries and structured feedback reports. You consolidate multiple reviews "
        "into one document, highlight the most critical issues, and provide "
        "actionable recommendations the team can act on immediately."
    ),
    llm=llm,
    verbose=True,
)

# --- Tasks ---

parse_task = Task(
    agent=prd_parser,
    name="Parse PRD Document",
    description=(
        "Read the PRD file at path: {prd_path}. "
        "Extract and structure all key sections: title, problem statement, objectives, "
        "user personas, user stories, functional requirements, non-functional "
        "requirements, out-of-scope items, success metrics, and open questions. "
        "Present the content in a clean structured format."
    ),
    expected_output=(
        "A structured extraction of all PRD sections with their full content, "
        "clearly labelled and ready for specialist review."
    ),
)

tech_task = Task(
    context=[parse_task],
    agent=tech_reviewer,
    name="Technical Feasibility Review",
    description=(
        "Using the parsed PRD content, conduct a thorough technical review. "
        "Assess: (1) technical feasibility of each requirement, "
        "(2) ambiguous or conflicting specs, "
        "(3) missing technical details (APIs, data models, performance targets), "
        "(4) integration and dependency risks, "
        "(5) scalability and security considerations. "
        "Rate overall technical readiness as: Ready / Needs Work / Major Issues."
    ),
    expected_output=(
        "A technical review with numbered findings, each with a severity label "
        "(Critical / Major / Minor), a description, and a recommended fix. "
        "Conclude with an overall readiness rating and top 3 action items."
    ),
)

ux_task = Task(
    context=[parse_task],
    agent=ux_reviewer,
    name="UX and Product Review",
    description=(
        "Using the parsed PRD content, conduct a UX and product quality review. "
        "Assess: (1) clarity of the problem statement and user need, "
        "(2) quality and completeness of user stories and acceptance criteria, "
        "(3) edge cases and error states that are missing, "
        "(4) whether the proposed solution is the right one for the stated problem, "
        "(5) accessibility and inclusivity considerations. "
        "Rate overall UX readiness as: Ready / Needs Work / Major Issues."
    ),
    expected_output=(
        "A UX review with numbered findings, each with a severity label "
        "(Critical / Major / Minor), a description, and a recommended fix. "
        "Conclude with an overall readiness rating and top 3 action items."
    ),
)

business_task = Task(
    context=[parse_task],
    agent=business_reviewer,
    name="Business Strategy Review",
    description=(
        "Using the parsed PRD content, conduct a business strategy review. "
        "Assess: (1) clarity of the business opportunity and target market, "
        "(2) competitive differentiation, "
        "(3) whether success metrics are measurable and tied to business outcomes, "
        "(4) assumptions that need validation, "
        "(5) revenue, cost, or growth impact. "
        "Rate overall business readiness as: Ready / Needs Work / Major Issues."
    ),
    expected_output=(
        "A business review with numbered findings, each with a severity label "
        "(Critical / Major / Minor), a description, and a recommended fix. "
        "Conclude with an overall readiness rating and top 3 action items."
    ),
)

risk_task = Task(
    context=[parse_task, tech_task, ux_task, business_task],
    agent=risk_assessor,
    name="Risk and Gaps Assessment",
    description=(
        "Using the parsed PRD and all specialist reviews, identify cross-cutting risks "
        "and gaps that were not fully addressed. Focus on: "
        "(1) security, privacy, and compliance gaps, "
        "(2) missing rollout and migration strategy, "
        "(3) unresolved open questions that block engineering, "
        "(4) dependency or third-party risks, "
        "(5) any single points of failure in the plan."
    ),
    expected_output=(
        "A risk register with each risk described, its likelihood and impact "
        "(High / Medium / Low), and a mitigation recommendation. "
        "Include a list of open questions that must be answered before development starts."
    ),
)

report_task = Task(
    context=[tech_task, ux_task, business_task, risk_task],
    agent=report_writer,
    name="Consolidated PRD Review Report",
    description=(
        "Synthesize all specialist reviews into one structured PRD Review Report. "
        "The report must include: "
        "(1) Executive Summary — overall verdict and top 3 blockers, "
        "(2) Technical Review Summary, "
        "(3) UX & Product Review Summary, "
        "(4) Business Strategy Review Summary, "
        "(5) Risk Register Summary, "
        "(6) Prioritised Action Items table (Owner, Priority, Action), "
        "(7) Recommended next steps before the PRD is approved. "
        "Be direct and actionable. Avoid vague language."
    ),
    expected_output=(
        "A complete, well-structured PRD Review Report in Markdown format "
        "that a product team can use immediately to improve the document."
    ),
    output_file="prd_review_report.md",
)

# --- Crew ---

crew = Crew(
    name="PRD Review Crew",
    agents=[prd_parser, tech_reviewer, ux_reviewer, business_reviewer, risk_assessor, report_writer],
    tasks=[parse_task, tech_task, ux_task, business_task, risk_task, report_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    prd_path = input("Enter the path to your PRD file (e.g. prd.md): ").strip() or PRD_FILE_PATH
    result = crew.kickoff(inputs={"prd_path": prd_path})
    print("\n" + "=" * 60)
    print("PRD REVIEW COMPLETE — report saved to prd_review_report.md")
    print("=" * 60)
    print(result.raw)
