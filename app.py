# =====================================================
# STEP 1: IMPORT LIBRARIES
# =====================================================
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os
import time
import pandas as pd
from dotenv import load_dotenv


# =====================================================
# STEP 2: LOAD API KEY
# =====================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")


# =====================================================
# STEP 3: MODEL CONFIGURATION
# =====================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=OPENAI_API_KEY
)


# =====================================================
# STEP 4: DEFINE TICKET MODEL
# =====================================================
class Ticket(BaseModel):
    category: str
    priority: str
    title: str
    details: str
    technical_details: str


# =====================================================
# STEP 5: CREATE OUTPUT FOLDER
# =====================================================
os.makedirs("output", exist_ok=True)


# =====================================================
# STEP 6: LOAD CSV DATA
# =====================================================
def load_all_feedback():
    reviews = pd.read_csv("feedback_data/reviews.csv")
    emails = pd.read_csv("feedback_data/emails.csv")

    reviews["source_type"] = "review"
    emails["source_type"] = "email"

    reviews.rename(columns={
        "review_id": "source_id",
        "review_text": "text"
    }, inplace=True)

    emails.rename(columns={
        "email_id": "source_id",
        "body": "text"
    }, inplace=True)

    combined = pd.concat([
        reviews[["source_id", "source_type", "text"]],
        emails[["source_id", "source_type", "text"]]
    ])

    return combined


# =====================================================
# STEP 7: DEFINE AGENTS
# =====================================================
classifier_agent = Agent(
    role="Feedback Classifier",
    goal="Classify feedback into Bug, Feature Request, Praise, Complaint, Spam",
    backstory="Expert AI engineer specialized in NLP classification.",
    llm=llm,
    verbose=False
)



ticket_creator_agent = Agent(
    role="Ticket Creator",
    goal="Generate structured engineering tickets",
    backstory="Expert product manager creating actionable engineering tickets.",
    llm=llm,
    verbose=False
)

quality_agent = Agent(
    role="Quality Critic",
    goal="Ensure tickets are complete and accurate",
    backstory="Senior QA engineer ensuring ticket quality.",
    llm=llm,
    verbose=False
)


# =====================================================
# STEP 8: CREATE TASKS
# =====================================================
def create_tasks(feedback):

    classify_task = Task(
        description=f"""
        Classify this feedback:
        {feedback}

        Categories: Bug, Feature Request, Praise, Complaint, Spam
        Assign priority: Critical, High, Medium, Low

        Return ONLY JSON:
        {{
            "category": "",
            "priority": ""
        }}
        """,
        agent=classifier_agent,
        expected_output="JSON with category and priority"
    )

    

    ticket_task = Task(
    description=f"""
    You are creating an engineering ticket from user feedback.

    Use the classification result from the previous task.

    Rules:

    1. Use the category and priority from the classification task.
    2. Generate a clear short title for the ticket.
    3. Write detailed description in "details".
    4. If category == "Bug":
        - Infer possible technical issue from the feedback.
        - Provide useful debugging hints in "technical_details".
        - Example: module affected, possible cause, error scenario.
    5. If category != "Bug":
        - Set "technical_details" to "" (empty string).
    6. Do NOT invent facts that are not implied in the feedback.

    Return STRICT JSON only:

    {{
        "category": "",
        "priority": "",
        "title": "",
        "details": "",
        "technical_details": ""
    }}

    Original Feedback:
    {feedback}

    Return ONLY JSON.
    """,
    agent=ticket_creator_agent,
    context=[classify_task],
    expected_output="Structured JSON Ticket"
)

    quality_task = Task(
        description="""
        Validate and correct the final ticket JSON.

        RULES:
        - Ensure valid JSON only.
        - No extra text.
        - All required fields must exist.
        - Category & priority must match classification.
        - If category != Bug → technical_details must be "".
        - Do not hallucinate information.

        Return ONLY the final corrected JSON.
        """,
        agent=quality_agent,
        context=[classify_task, ticket_task],
        expected_output="Final validated JSON ticket",
        output_pydantic=Ticket
    )

    return [classify_task, ticket_task, quality_task]


# =====================================================
# STEP 9: PROCESS FEEDBACK
# =====================================================
def process_feedback(source_id, source_type, feedback):
    try:
        tasks = create_tasks(feedback)

        crew = Crew(
            agents=[
                classifier_agent,
                
                ticket_creator_agent,
                quality_agent
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=False
        )

        crew_output = crew.kickoff()

        final_ticket = crew_output.tasks_output[-1].pydantic

        return {
            "source_id": source_id,
            "source_type": source_type,
            "category": final_ticket.category,
            "priority": final_ticket.priority,
            "title": final_ticket.title,
            "details": final_ticket.details,
            "technical_details": final_ticket.technical_details
        }

    except Exception as e:
        print(f"Error processing feedback {source_id}: {e}")

        return {
            "source_id": source_id,
            "source_type": source_type,
            "category": "Error",
            "priority": "Low",
            "title": "Processing Failed",
            "details": feedback,
            "technical_details": ""
        }


# =====================================================
# STEP 10: LOG METRICS
# =====================================================
def log_metrics(total_processed):
    file = "output/metrics.csv"

    metrics = pd.DataFrame([{
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_processed": total_processed
    }])

    if not os.path.exists(file):
        metrics.to_csv(file, index=False)
    else:
        metrics.to_csv(file, mode="a", header=False, index=False)


# =====================================================
# STEP 11: STREAMLIT UI
# =====================================================
def run_ui():
    st.title("CrewAI Feedback Analysis System")

    if "processed" not in st.session_state:
        st.session_state.processed = False

    if st.button("Start Processing"):

        if os.path.exists("output/generated_tickets.csv"):
            os.remove("output/generated_tickets.csv")

        if os.path.exists("output/metrics.csv"):
            os.remove("output/metrics.csv")

        data = load_all_feedback()
        total = len(data)

        progress = st.progress(0)
        tickets = []

        for i, row in data.iterrows():
            result = process_feedback(
                row["source_id"],
                row["source_type"],
                row["text"]
            )

            tickets.append(result)
            progress.progress((i + 1) / total)

        df = pd.DataFrame(tickets)
        df.to_csv("output/generated_tickets.csv", index=False)

        log_metrics(total)

        st.session_state.processed = True
        st.success("Processing Completed")

    if st.session_state.processed:

        if os.path.exists("output/generated_tickets.csv"):
            st.subheader("Generated Tickets")
            df = pd.read_csv("output/generated_tickets.csv")
            st.dataframe(df)

        if os.path.exists("output/metrics.csv"):
            st.subheader("Metrics")
            df = pd.read_csv("output/metrics.csv")
            st.dataframe(df)


# =====================================================
# STEP 12: MAIN
# =====================================================
if __name__ == "__main__":
    run_ui()