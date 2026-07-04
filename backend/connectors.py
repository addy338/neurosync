import json
import random
import urllib.request
import urllib.error
import subprocess
import os
from typing import Tuple

import google.generativeai as genai

def execute_python_code(code: str) -> str:
    """Executes the given Python code locally and returns its stdout/stderr. Use this to perform calculations, data processing, or fetching data."""
    try:
        result = subprocess.run(["python", "-c", code], capture_output=True, text=True, timeout=15)
        out = result.stdout + "\n" + result.stderr
        return out if out.strip() else "Executed successfully with no output."
    except Exception as e:
        return f"Execution Error: {e}"

class CloudConnectors:
    """
    Real connector for Cloud AI APIs (Gemini)
    """
    def __init__(self):
        # Configure Gemini using the key from .env
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash', 
            tools=[execute_python_code],
            system_instruction="You are NeuroSync, the Omni-AI Orchestrator. When given a complex task (like math, data processing, fetching web data), you MUST write and execute Python code using your execute_python_code tool to find the answer. Do not guess. Once you get the result from the tool, synthesize a beautiful, well-formatted Markdown response."
        )

    def query(self, prompt: str) -> Tuple[str, str]:
        """
        Sends the prompt to Google Gemini.
        """
        try:
            chat = self.model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(prompt)
            return "Orchestrator-Node", response.text
        except Exception as e:
            return "System-Error", f"Failed to connect to Gemini: {str(e)}"

class LocalConnectors:
    """
    Real connector for Local AI APIs (Ollama, Python Executor)
    """
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"

    def execute_python(self, prompt: str) -> Tuple[str, str]:
        """
        Executes basic math logic as the Python node.
        """
        try:
            clean_expr = "".join(c for c in prompt if c in "0123456789+-*/().")
            if clean_expr:
                ans = eval(clean_expr)
                return "Python-Executor", f"Calculated Result: {ans}"
            return "Python-Executor", "I am a local Python Executor. Give me a math problem like 'calculate 5 * 10'."
        except Exception as e:
            return "Python-Executor", f"Error during execution: {e}"

    def query(self, prompt: str) -> Tuple[str, str]:
        """
        Attempts to route the prompt to Ollama.
        If Ollama is down, falls back to local Python execution for basic commands.
        """
        # Try Ollama (using llama3.2:latest)
        try:
            data = json.dumps({
                "model": "llama3.2:latest",
                "prompt": prompt,
                "stream": False
            }).encode('utf-8')
            
            req = urllib.request.Request(self.ollama_url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=3) as response:
                result = json.loads(response.read().decode('utf-8'))
                return "Ollama-Llama3.2", result.get("response", "No response text")
        except Exception as e:
            print(f"Ollama connection failed: {e}. Falling back to Python Executor.")

        # Fallback
        return self.execute_python(prompt)
