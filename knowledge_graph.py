import os
import json
from neo4j import GraphDatabase
from utils import load_environment


def get_neo4j_driver():
    """
    Initializes and returns the Neo4j driver.
    Expects NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD to be set in the environment.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, user, password]):
        raise ValueError("Neo4j connection details not found in environment variables.")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        raise

# --- GRAPH UPDATE LOGIC ---

def _add_relationship_tx(tx, source, relationship, target):
    """
    A transactional function to create nodes and relationships in Neo4j.
    Uses MERGE to avoid creating duplicate entities.
    
    Args:
        tx: The Neo4j transaction object.
        source (str): The name of the source entity.
        relationship (str): The type of the relationship.
        target (str): The name of the target entity.
    """
    # Use MERGE to find or create the source node. We'll give it a generic 'Entity' label.
    # You could enhance this to use different labels for different entity types if known.
    tx.run(
        "MERGE (s:Entity {name: $source_name})",
        source_name=source
    )
    # MERGE the target node
    tx.run(
        "MERGE (t:Entity {name: $target_name})",
        target_name=target
    )
    # MERGE the relationship between the source and target nodes
    query = """
    MATCH (s:Entity {name: $source_name})
    MATCH (t:Entity {name: $target_name})
    MERGE (s)-[r:%s]->(t)
    """ % relationship.upper().replace(" ", "_") # Sanitize relationship type for Cypher
    
    tx.run(query, source_name=source, target_name=target)

def update_graph_from_json(driver, json_data, source_document_id):
    """
    Parses a JSON string with entities/relationships and updates the graph.
    
    Args:
        driver: The Neo4j driver instance.
        json_data (str): A JSON string from the LLM, e.g.,
                         '[{"source": "John", "relationship": "WORKS_FOR", "target": "Acme"}]'
        source_document_id (str): An identifier for the source document (e.g., filename).
    """
    try:
        data = json.loads(json_data)
        if not isinstance(data, list):
            print("Warning: JSON data is not a list. Skipping graph update.")
            return

        with driver.session() as session:
            for item in data:
                source = item.get("source")
                relationship = item.get("relationship")
                target = item.get("target")

                if all([source, relationship, target]):
                    print(f"Adding to graph: {source} -> {relationship} -> {target}")
                    session.execute_write(_add_relationship_tx, source, relationship, target)
                else:
                    print(f"Skipping malformed item: {item}")
        
        print("Graph update complete.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from LLM output: {json_data}")
    except Exception as e:
        print(f"An error occurred while updating the graph: {e}")

# --- GRAPH QUERYING ---

def query_graph(driver, cypher_query):
    """
    Executes a read-only Cypher query against the graph.
    
    Args:
        driver: The Neo4j driver instance.
        cypher_query (str): The Cypher query to execute.
        
    Returns:
        list: A list of records from the query result.
    """
    try:
        with driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]
    except Exception as e:
        print(f"An error occurred during graph query: {e}")
        return []

# --- MAIN DRIVER LIFECYCLE ---

def close_driver(driver):
    """Closes the Neo4j driver connection."""
    if driver:
        driver.close()
        print("Neo4j driver closed.")

# Example usage (for testing purposes)
if __name__ == '__main__':
    load_environment()
    neo4j_driver = None
    try:
        neo4j_driver = get_neo4j_driver()
        
        # Example data from an LLM
        example_json = """
        [
          {"source": "Project Phoenix", "relationship": "IS_MANAGED_BY", "target": "Alice"},
          {"source": "Alice", "relationship": "WORKS_FOR", "target": "Innovate Inc."},
          {"source": "Project Phoenix", "relationship": "IS_A", "target": "RAG System"}
        ]
        """
        
        update_graph_from_json(neo4j_driver, example_json, "doc1.pdf")
        
        # Example query
        results = query_graph(neo4j_driver, "MATCH (n) RETURN n.name AS name LIMIT 10")
        print("Query results:", results)
        
    finally:
        close_driver(neo4j_driver)