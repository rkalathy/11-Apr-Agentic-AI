import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.flow.flow import start, listen, Flow, router, or_
from pydantic import BaseModel


class CustomerCareState(BaseModel):
    customer_name: str = ""
    issue: str = ""
    issue_category: str = ""  # billing, technical, general
    resolution: str = ""
    ticket_summary: str = ""


load_dotenv()
llm = LLM(model="openai/gpt-4o-mini", temperature=0)
llm2 = LLM(model="openai/gpt-4o-mini", temperature=0.5)


class CustomerCareFlow(Flow[CustomerCareState]):

    @start()
    def intake_issue(self):
        print(f"\nCustomer: {self.state.customer_name}")
        print(f"Issue: {self.state.issue}")

        classifier_agent = Agent(
            role="Issue Classifier",
            goal="Accurately classify the customer's issue into the correct category",
            backstory=(
                "You are a skilled customer service triage specialist who quickly "
                "determines whether issues are related to billing, technical problems, "
                "or general inquiries."
            ),
            llm=llm,
            verbose=True,
        )
        task = Task(
            agent=classifier_agent,
            description=(
                f"Classify the following customer issue into exactly one of these categories: "
                f"'billing', 'technical', or 'general'.\n\n"
                f"Customer issue: {self.state.issue}\n\n"
                "Respond with ONLY the category word: billing, technical, or general."
            ),
            expected_output="A single word: billing, technical, or general",
        )
        crew = Crew(agents=[classifier_agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.issue_category = result.raw.strip().lower()
        print(f"Issue classified as: {self.state.issue_category}")

    @router(intake_issue)
    def route_to_specialist(self):
        if self.state.issue_category in ["billing", "technical", "general"]:
            return self.state.issue_category
        return "general"  # default fallback

    @listen("billing")
    def billing_specialist(self):
        agent = Agent(
            role="Billing Specialist",
            goal="Resolve billing inquiries and payment-related issues for customers",
            backstory=(
                "You are a meticulous billing specialist with deep knowledge of "
                "invoicing, payment processing, subscription management, and refund policies. "
                "You handle disputes with empathy and accuracy."
            ),
            llm=llm2,
            verbose=True,
        )
        task = Task(
            agent=agent,
            description=(
                f"A customer named {self.state.customer_name} has a billing issue:\n"
                f"{self.state.issue}\n\n"
                "Diagnose the issue and provide a detailed resolution with specific steps. "
                "Include any relevant billing policies, refund options, or payment adjustments."
            ),
            expected_output=(
                "A detailed resolution for the billing issue including diagnosis, "
                "recommended actions, and any applicable policies."
            ),
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.resolution = result.raw

    @listen("technical")
    def technical_specialist(self):
        agent = Agent(
            role="Technical Support Engineer",
            goal="Diagnose and resolve technical issues customers are experiencing",
            backstory=(
                "You are a senior technical support engineer with expertise in "
                "troubleshooting software, hardware, connectivity, and account access issues. "
                "You provide clear step-by-step solutions."
            ),
            llm=llm2,
            verbose=True,
        )
        task = Task(
            agent=agent,
            description=(
                f"A customer named {self.state.customer_name} has a technical issue:\n"
                f"{self.state.issue}\n\n"
                "Diagnose the root cause and provide step-by-step troubleshooting instructions. "
                "Include any known workarounds or escalation paths if needed."
            ),
            expected_output=(
                "A step-by-step technical resolution including root cause analysis, "
                "troubleshooting steps, and escalation options if required."
            ),
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.resolution = result.raw

    @listen("general")
    def general_support(self):
        agent = Agent(
            role="Customer Support Representative",
            goal="Handle general customer inquiries and provide helpful, accurate information",
            backstory=(
                "You are a friendly and knowledgeable customer support representative "
                "who handles general questions about products, services, policies, and "
                "account management. You always aim for first-contact resolution."
            ),
            llm=llm2,
            verbose=True,
        )
        task = Task(
            agent=agent,
            description=(
                f"A customer named {self.state.customer_name} has the following inquiry:\n"
                f"{self.state.issue}\n\n"
                "Provide a helpful, accurate, and complete response. "
                "Include relevant product/service information and any next steps."
            ),
            expected_output=(
                "A comprehensive and friendly response addressing the customer's inquiry "
                "with all relevant information and clear next steps."
            ),
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.resolution = result.raw

    @listen(or_(billing_specialist, technical_specialist, general_support))
    def ticket_writer(self):
        agent = Agent(
            role="Ticket Documentation Specialist",
            goal="Create accurate and complete support ticket summaries",
            backstory=(
                "You are a detail-oriented documentation specialist who creates clear "
                "support ticket summaries for quality review and customer records."
            ),
            llm=llm,
            verbose=True,
        )
        task = Task(
            agent=agent,
            description=(
                f"Create a support ticket summary for the following interaction:\n\n"
                f"Customer: {self.state.customer_name}\n"
                f"Category: {self.state.issue_category}\n"
                f"Issue: {self.state.issue}\n"
                f"Resolution: {self.state.resolution}\n\n"
                "Format as a structured ticket with: Ticket ID (generate one), Customer, "
                "Category, Issue Summary, Resolution, Status, and Follow-up Required (yes/no)."
            ),
            expected_output=(
                "A structured support ticket summary with all required fields completed."
            ),
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.ticket_summary = result.raw


flow = CustomerCareFlow()
result = flow.kickoff(inputs={
    "customer_name": "Alice Johnson",
    "issue": "I was charged twice for my subscription this month and need a refund."
})
print("\n=== TICKET SUMMARY ===")
print(result.ticket_summary)
