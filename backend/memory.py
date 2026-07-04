import chromadb
import os
import uuid

# Initialize ChromaDB. 
# We use a persistent client so memories survive server restarts.
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
client = chromadb.PersistentClient(path=DB_DIR)

# Get or create the collection for storing chat memories
# We use the default embedding function (all-MiniLM-L6-v2) which is lightweight and fast
try:
    collection = client.get_or_create_collection(name="neurosync_memory")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    collection = None

class RAGMemoryManager:
    """
    Handles Long-Term Memory for the Hive.
    Converts conversations into vector embeddings and stores them.
    Retrieves the most semantically relevant past conversations.
    """
    def __init__(self):
        self.collection = collection

    # Bug #10 fix: node names / response prefixes that indicate a failed
    # call rather than a real answer — these must never be recalled later
    # as if they were genuine past context.
    _ERROR_NODE_MARKERS = ("System-Error", "Ollama-Offline")

    def add_memory(self, prompt: str, response: str, model_used: str):
        """Stores a prompt-response pair as a memory vector."""
        if not self.collection:
            return
        if any(marker in model_used for marker in self._ERROR_NODE_MARKERS):
            return
        if response.strip().startswith("⚠️"):
            return

        memory_id = str(uuid.uuid4())
        
        # The document is the full context we want the AI to remember
        document = f"User asked: {prompt}\nAI ({model_used}) replied: {response}"
        
        # Metadata allows us to filter later if we want
        metadata = {"prompt": prompt, "model": model_used}
        
        try:
            self.collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[memory_id]
            )
            print(f"🧠 Memory stored: {memory_id}")
        except Exception as e:
            print(f"Failed to store memory: {e}")

    def recall_memory(self, query: str, n_results: int = 2) -> str:
        """
        Searches the vector database for memories semantically related to the query.
        Returns a formatted string of past context to inject into the prompt.
        """
        if not self.collection:
            return ""
            
        try:
            count = self.collection.count()
            if count == 0:
                return ""
            
            # Query ChromaDB. It automatically embeds the query and finds the closest vectors.
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, count)
            )
            
            documents = results.get("documents", [[]])[0]
            if not documents:
                return ""
                
            # Format the retrieved memories into a context block
            context = "--- PAST CONTEXT (Use this if relevant) ---\n"
            for i, doc in enumerate(documents):
                context += f"Memory {i+1}:\n{doc}\n\n"
            context += "-------------------------------------------\n\n"
            
            return context
            
        except Exception as e:
            print(f"Failed to recall memory: {e}")
            return ""

# Export a singleton instance
memory = RAGMemoryManager()
