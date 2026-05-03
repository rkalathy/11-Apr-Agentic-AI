# Customer Care Flow — CrewAI Example

A CrewAI Flow example that routes incoming customer issues to the right specialist agent using `@router` and string-based `@listen` branching.

## Flow Architecture

```
intake_issue (@start)
     │  LLM classifies issue into: billing | technical | general
     ▼
route_to_specialist (@router)
     │  returns a string matching the category
     ├──→ billing_specialist   (@listen("billing"))
     ├──→ technical_specialist (@listen("technical"))
     └──→ general_support      (@listen("general"))
                │  only one branch runs per request
                ▼
          ticket_writer (@listen(or_(...)))
               writes a structured support ticket
```

## Agents

| Agent | Triggered by | Responsibility |
|---|---|---|
| Issue Classifier | `@start` | Classifies the customer's issue into `billing`, `technical`, or `general` |
| Billing Specialist | `@listen("billing")` | Resolves payment, invoice, subscription, and refund issues |
| Technical Support Engineer | `@listen("technical")` | Diagnoses and troubleshoots software/hardware/connectivity problems |
| Customer Support Representative | `@listen("general")` | Handles general product, policy, and account inquiries |
| Ticket Documentation Specialist | `@listen(or_(...))` | Summarises the interaction into a structured support ticket |

## State

```python
class CustomerCareState(BaseModel):
    customer_name: str  # name of the customer
    issue: str          # raw issue description from the customer
    issue_category: str # set by classifier: billing | technical | general
    resolution: str     # set by the specialist agent
    ticket_summary: str # set by the ticket writer
```

## Setup

```bash
pip install crewai python-dotenv
```

Add your OpenAI key to a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## Usage

Run with a billing issue (default):

```bash
python customer_care_flow.py
```

To test a different branch, edit the `kickoff` call at the bottom of the file:

```python
# Billing branch
flow.kickoff(inputs={
    "customer_name": "Alice Johnson",
    "issue": "I was charged twice for my subscription this month and need a refund."
})

# Technical branch
flow.kickoff(inputs={
    "customer_name": "Bob Smith",
    "issue": "The app crashes every time I try to log in on my iPhone."
})

# General branch
flow.kickoff(inputs={
    "customer_name": "Carol White",
    "issue": "What are your cancellation and refund policies?"
})
```

## Key Concepts

- **`@router`** — inspects state after `intake_issue` and returns a string (`"billing"`, `"technical"`, or `"general"`)
- **`@listen("string")`** — only fires when the router returns that exact string; the other two specialist methods stay idle
- **`or_(...)`** — merges all three specialist branches back into a single `ticket_writer` step regardless of which branch ran
