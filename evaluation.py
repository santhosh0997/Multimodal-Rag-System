import pytest
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

# Import our custom modules
import utils
import vector_db
import knowledge_graph
import hybrid_search
import ingestion


utils.load_environment()

QDRANT_COLLECTION_NAME = "evaluation_collection"
qdrant_client = vector_db.get_qdrant_client()
neo4j_driver = knowledge_graph.get_neo4j_driver()

# Define the models for evaluation
relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4", include_reason=True)
faithfulness_metric = FaithfulnessMetric(threshold=0.7, model="gpt-4", include_reason=True)
contextual_recall_metric = ContextualRecallMetric(threshold=0.7, model="gpt-4", include_reason=True)

def setup_test_environment():
    """
    Sets up the database with a known set of data for consistent testing.
    This function should be run once before the evaluation.
    """
    print("--- Setting up test environment ---")
    
    # Ensure the Qdrant collection exists and is clean (optional, but good practice)
    vector_db.create_collection_if_not_exists(qdrant_client, QDRANT_COLLECTION_NAME)
    
    # Create a dummy file with our test data
    test_data = """
    Dr. Evelyn Reed, a leading researcher at Innovate Dynamics, published a paper on Quantum RAG systems.
    The paper, titled 'Quantum Leaps in AI', was co-authored by Dr. Ben Carter.
    Innovate Dynamics is headquartered in Geneva. The project code for this research is QR-7.
    """
    test_file_path = "test_document.txt"
    with open(test_file_path, "w") as f:
        f.write(test_data)
        
    # Ingest this file into our test collection
    ingestion.ingest_file(
        file_path=test_file_path,
        collection_name=QDRANT_COLLECTION_NAME,
        qdrant_client=qdrant_client,
        neo4j_driver=neo4j_driver
    )
    print("--- Test environment setup complete ---")

def run_rag_pipeline(query: str):
    """
    A helper function to run the full RAG pipeline for a given query.
    Returns the final answer and the retrieved context.
    """
    context = hybrid_search.hybrid_retrieval(
        query, QDRANT_COLLECTION_NAME, qdrant_client, neo4j_driver
    )
    answer = hybrid_search.generate_response(query, context)
    return answer, context

# --- TEST CASES DEFINITION ---

# 1. Factual Lookup Query
query1 = "Who published the paper on Quantum RAG systems?"
answer1, context1 = run_rag_pipeline(query1)
test_case_1 = LLMTestCase(
    input=query1,
    actual_output=answer1,
    expected_output="Dr. Evelyn Reed published the paper on Quantum RAG systems.",
    retrieval_context=[context1],
    context=[context1] # For faithfulness and recall
)

# 2. Relationship / Semantic Linkage Query
query2 = "Who works for Innovate Dynamics?"
answer2, context2 = run_rag_pipeline(query2)
test_case_2 = LLMTestCase(
    input=query2,
    actual_output=answer2,
    expected_output="Dr. Evelyn Reed works at Innovate Dynamics.",
    retrieval_context=[context2],
    context=[context2]
)

# 3. Summarization / Multi-hop Query
query3 = "Where is the company that Dr. Ben Carter is associated with located?"
answer3, context3 = run_rag_pipeline(query3)
test_case_3 = LLMTestCase(
    input=query3,
    actual_output=answer3,
    # This answer requires connecting Ben Carter -> Paper -> Evelyn Reed -> Innovate Dynamics -> Geneva
    expected_output="The company Dr. Ben Carter is associated with, Innovate Dynamics, is located in Geneva.",
    retrieval_context=[context3],
    context=[context3]
)

# --- PYTEST EXECUTION ---
@pytest.mark.parametrize(
    "test_case",
    [test_case_1, test_case_2, test_case_3]
)
def test_rag_system(test_case: LLMTestCase):
    """
    This is the actual test function that pytest will discover and run.
    It evaluates each test case against our defined metrics.
    """
    evaluate(
        test_cases=[test_case],
        metrics=[relevancy_metric, faithfulness_metric, contextual_recall_metric]
    )

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("Running evaluation suite...")
    # 1. Setup the data
    setup_test_environment()
    
    # 2. Define all test cases
    # We re-run the pipeline here because the setup might have changed the state
    print("\n--- Generating outputs for test cases ---")
    query1 = "Who published the paper on Quantum RAG systems?"
    answer1, context1 = run_rag_pipeline(query1)
    tc1 = LLMTestCase(input=query1, actual_output=answer1, retrieval_context=[context1], context=[context1], expected_output="Dr. Evelyn Reed published the paper.")

    query2 = "What is the project code for the Quantum RAG research?"
    answer2, context2 = run_rag_pipeline(query2)
    tc2 = LLMTestCase(input=query2, actual_output=answer2, retrieval_context=[context2], context=[context2], expected_output="The project code is QR-7.")
    
    all_test_cases = [tc1, tc2]

    # 3. Run the evaluation
    print("\n--- Starting DeepEval Evaluation ---")
    evaluation_results = evaluate(
        test_cases=all_test_cases,
        metrics=[relevancy_metric, faithfulness_metric, contextual_recall_metric],
        print_results=True,
    )
    
    print("\n--- Evaluation complete. Tearing down. ---")
    knowledge_graph.close_driver(neo4j_driver)