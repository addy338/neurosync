import json
import random
import urllib.request
import urllib.error
import subprocess
from typing import Tuple

class CloudConnectors:
    """
    Mock connector for Cloud AI APIs (Gemini, Claude)
    """
    def __init__(self):
        self.available_models = ["gemini-pro", "claude-3-opus"]

    def query(self, prompt: str) -> str:
        model = random.choice(self.available_models)
        return f"[{model}] Successfully processed: {prompt[:20]}..."

class LocalConnectors:
    """
    Real connector for Local AI APIs (Ollama, Python Executor)
    """
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"

    def query(self, prompt: str) -> Tuple[str, str]:
        """
        Attempts to route the prompt to Ollama.
        If Ollama is down, falls back to local Python execution for basic commands.
        """
        # 1. Try Ollama (assuming llama3 is installed)
        try:
            data = json.dumps({
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }).encode('utf-8')
            
            req = urllib.request.Request(self.ollama_url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=3) as response:
                result = json.loads(response.read().decode('utf-8'))
                return "Ollama-Llama3", result.get("response", "No response text")
        except Exception as e:
            print(f"Ollama connection failed: {e}. Falling back to Python Executor.")

        # 2. Fallback: Python Executor (very basic execution)
        if prompt.lower().startswith("echo"):
            return "Python-Executor", prompt[5:]
        elif "calculate" in prompt.lower() or "+" in prompt or "*" in prompt:
            # Dangerous in prod, but a cool local demo
            try:
                # Strip text, try to just evaluate the math
                clean_expr = "".join(c for c in prompt if c in "0123456789+-*/().")
                if clean_expr:
                    ans = eval(clean_expr)
                    return "Python-Executor", f"Calculated Result: {ans}"
            except:
                pass
        
        return "System-Node", f"Received your prompt: '{prompt}'. (Ollama is not running locally, so I used the fallback node to just echo this back!)"
