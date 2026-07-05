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

        # Never store failed/error/execution-failure responses as if they
        # were real answers — otherwise future prompts recall them as "past
        # context" and confuse whichever model reads them back.
        if model_used in ("System-Error", "Ollama-Offline"):
            return
        if response.strip().startswith("⚠️"):
            return
        # Broadened: also catch code-execution failures regardless of which
        # node produced them (Python-Executor, Gemini tool-calling, etc.).
        # These don't come from a System-Error node and don't start with ⚠️,
        # so the earlier filter let them through into the vector store.
        _failure_markers = (
            "SyntaxError",
            "Execution Error",
            "Traceback (most recent call last)",
            "Could not evaluate expression",
            "NameError",
            "TypeError:",
            "only evaluates arithmetic expressions",
        )
        if any(marker in response for marker in _failure_markers):
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

    def clear_all(self) -> bool:
        """
        Wipes every stored memory from the vector database.
        Use after fixing the error filter (so a fresh start doesn't get
        immediately re-poisoned), or any time old/bad memories need to be
        purged without restarting the server or touching files by hand.

        Returns True on success, False if the collection isn't available
        or the delete call fails.
        """
        if not self.collection:
            return False
        try:
            all_ids = self.collection.get().get("ids", [])
            if all_ids:
                self.collection.delete(ids=all_ids)
                print(f"🧠 Memory cleared: {len(all_ids)} entries deleted")
            else:
                print("🧠 Memory already empty — nothing to clear")
            return True
        except Exception as e:
            print(f"Failed to clear memory: {e}")
            return False

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
