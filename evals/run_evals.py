import os
import sys

# --- PATH FIX: Tell Python where the root folder is ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nest_asyncio
import numpy as np

# --- MODERN TRULENS IMPORTS ---
from trulens.core import TruSession, Feedback, Select
from trulens.apps.llamaindex import TruLlama
from trulens.providers.openai import OpenAI as fOpenAI
from trulens.feedback.templates import Groundedness

# We import your exact RAG service and settings from the app!
from app.services.advanced_rag_service import AdvancedRAGService
from app.core.config import settings

# 1. The Juggling Trick
nest_asyncio.apply()

# 2. Turn on the Grader
tru = TruSession()
tru.reset_database() 

# 3. Setup the Teacher AI
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://models.inference.ai.azure.com"

provider = fOpenAI(model_engine="gpt-4o-mini") 

# Instantiate Groundedness EARLY so our proxy function can use it
grounded = Groundedness(groundedness_provider=provider)

print("Building the grading rubrics...")

# =====================================================================
# --- THE PROXY FIX (Bypassing the TruLens Signature Bug) ---
# By defining explicit parameters here, we force the OTEL engine 
# to see the correct names instead of getting confused by *args.
# =====================================================================
def grade_context(question: str, context: str):
    return provider.context_relevance_with_cot_reasons(question, context)

def grade_groundedness(source: str, statement: str):
    return grounded.groundedness_measure_with_cot_reasons(source, statement)

def grade_answer(question: str, statement: str):
    return provider.relevance_with_cot_reasons(question, statement)


# --- THE RAG TRIAD OF METRICS (OTEL COMPLIANT) ---
context_selection = TruLlama.select_source_nodes().node.text

# Metric 1: Context Relevance 
f_qs_relevance = (
    Feedback(grade_context, name="Context Relevance")
    .on({
        "question": Select.RecordInput, 
        "context": context_selection
    }) 
    .aggregate(np.mean)
)

# Metric 2: Groundedness
f_groundedness = (
    Feedback(grade_groundedness, name="Groundedness")
    .on({
        "source": context_selection, 
        "statement": Select.RecordOutput
    })
    .aggregate(grounded.grounded_statements_aggregator)
)

# Metric 3: Answer Relevance
f_qa_relevance = (
    Feedback(grade_answer, name="Answer Relevance")
    .on({
        "question": Select.RecordInput,
        "statement": Select.RecordOutput
    })
)

print("Booting up the Advanced RAG Service...")
# 4. Bring in the "Student" (Your actual production AI service)
rag_service = AdvancedRAGService()

# 5. Attach the Security Camera
tru_recorder = TruLlama(
    rag_service.query_engine,
    app_id="Production_Docs_QnA_v1",
    feedbacks=[f_qa_relevance, f_qs_relevance, f_groundedness]
)

# 6. Load the Golden Dataset
eval_questions = []
with open('evals/eval_questions.txt', 'r') as file:
    for line in file:
        if line.strip(): 
            eval_questions.append(line.strip())

# 7. Take the Test!
print("Starting the exam. The Teacher AI is now grading...")
for question in eval_questions:
    print(f"\nAsking: {question}")
    with tru_recorder as recording:
        rag_service.query_engine.query(question)

print("\nExam finished! Compiling the report card...")

# 8. Open the Dashboard
tru.run_dashboard()