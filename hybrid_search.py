import json

import utils
import vector_db
import knowledge_graph


def _generate_graph_query(llm, user_query: str) -> str:
    """
    Uses an LLM to convert a natural language query into a Cypher query for Neo4j.
    
    Args:
        llm: An initialized language model instance.
        user_query (str): The user's question in natural language.
        
    Returns:
        str: A Cypher query string.
    """
    prompt = f"""
    You are an expert Cypher query generator. Your task is to convert a user's
    natural language question into a Cypher query for a Neo4j graph.
    The graph has a simple schema: all nodes have the label `Entity` and a `name` property.
    Relationships are dynamic and represented by their type.

    Example 1:
    User Query: "What is the relationship between Project Phoenix and Alice?"
    Cypher Query:
    MATCH (p1:Entity {{name: 'Project Phoenix'}})-[r]-(p2:Entity {{name: 'Alice'}})
    RETURN p1.name AS Source, type(r) AS Relationship, p2.name AS Target

    Example 2:
    User Query: "Who works for Innovate Inc.?"
    Cypher Query:
    MATCH (p1:Entity)-[r:WORKS_FOR]->(p2:Entity {{name: 'Innovate Inc.'}})
    RETURN p1.name AS Person, p2.name AS Company

    Example 3:
    User Query: "What projects is Alice involved in?"
    Cypher Query:
    MATCH (p:Entity {{name: 'Alice'}})-[]-(project:Entity)
    RETURN project.name AS Project

    Now, generate a Cypher query for the following user question:
    ---
    User Query: "{user_query}"
    ---
    Cypher Query:
    """
    
    try:
        response = llm.invoke(prompt)
        # Clean up the response to ensure it's just the query
        return response.content.strip().replace("```cypher", "").replace("```", "")
    except Exception as e:
        print(f"Error generating Cypher query: {e}")
        return ""



def hybrid_retrieval(user_query: str, collection_name: str, qdrant_client, neo4j_driver) -> str:
    """
    Performs hybrid retrieval from both the vector DB and the knowledge graph.
    
    Args:
        user_query (str): The user's question.
        collection_name (str): The Qdrant collection name.
        qdrant_client: An active Qdrant client instance.
        neo4j_driver: An active Neo4j driver instance.
        
    Returns:
        str: A combined context string from both retrieval sources.
    """
    print("\n--- Starting Hybrid Retrieval ---")
    
    # 1. Semantic Search from Vector DB
    print("Step 1: Performing semantic search...")
    semantic_results = vector_db.semantic_search(
        client=qdrant_client,
        collection_name=collection_name,
        query_text=user_query,
        limit=3 # Retrieve top 3 chunks
    )
    semantic_context = "\n".join([hit.payload['text'] for hit in semantic_results])
    print(f"Found {len(semantic_results)} semantic results.")
    
    # 2. Structured Search from Knowledge Graph
    print("Step 2: Performing structured search on knowledge graph...")
    llm = utils.get_llm()
    cypher_query = _generate_graph_query(llm, user_query)
    graph_context = ""
    
    if cypher_query:
        print(f"Generated Cypher Query: {cypher_query}")
        graph_results = knowledge_graph.query_graph(neo4j_driver, cypher_query)
        print(f"Found {len(graph_results)} results from graph.")
        if graph_results:
            # Format graph results into a readable string
            graph_context = "\n".join([json.dumps(record) for record in graph_results])
    else:
        print("Could not generate a valid Cypher query.")

    # 3. Combine contexts
    combined_context = f"""
    CONTEXT FROM SEMANTIC SEARCH:
    ---
    {semantic_context}
    ---
    
    CONTEXT FROM KNOWLEDGE GRAPH:
    ---
    {graph_context}
    ---
    """
    
    print("--- Hybrid Retrieval Complete ---")
    return combined_context



def generate_response(query: str, context: str) -> str:
    """
    Generates a final answer using the retrieved context.
    
    Args:
        query (str): The original user query.
        context (str): The combined context from hybrid retrieval.
        
    Returns:
        str: The final, synthesized answer.
    """
    llm = utils.get_llm()
    prompt = f"""
    You are an intelligent assistant. Your task is to answer the user's question
    based *only* on the provided context. If the context does not contain the answer,
    say "I do not have enough information to answer this question."

    USER'S QUESTION:
    {query}

    PROVIDED CONTEXT:
    {context}

    ANSWER:
    """
    
    try:
        answer = llm.invoke(prompt)
        return answer.content.strip()
    except Exception as e:
        print(f"Error during final response generation: {e}")
        return "Sorry, I encountered an error while generating the response."


if __name__ == '__main__':
    utils.load_environment()
    
    QDRANT_COLLECTION_NAME = "multimodal_rag_collection"
    TEST_QUERY = "Who is the CEO of Global Tech Inc.?"
    
    qdrant_client = None
    neo4j_driver = None
    try:
        # Initialize clients
        qdrant_client = vector_db.get_qdrant_client()
        neo4j_driver = knowledge_graph.get_neo4j_driver()

        print(f"Testing with query: '{TEST_QUERY}'")
        
        # Perform retrieval
        retrieved_context = hybrid_retrieval(
            TEST_QUERY, 
            QDRANT_COLLECTION_NAME, 
            qdrant_client, 
            neo4j_driver
        )
        
        print("\n--- Retrieved Context ---")
        print(retrieved_context)
        
        # Generate final response
        final_answer = generate_response(TEST_QUERY, retrieved_context)
        
        print("\n--- Final Answer ---")
        print(final_answer)
        
    except Exception as e:
        print(f"An error occurred during the test run: {e}")
    finally:
        if neo4j_driver:
            knowledge_graph.close_driver(neo4j_driver)