import json
import os
import subprocess
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 🔧 SHARED LOCAL TOOL
# This function is given to the Orchestrator as
# a callable tool. It runs Python code on the
# LOCAL machine — something no cloud AI can do
# on its own. This is NeuroSync's superpower.
# ─────────────────────────────────────────────
def execute_python_code(code: str) -> str:
    """
    Executes the given Python code locally and returns stdout/stderr.
    Use this for calculations, data processing, or any task requiring
    deterministic computation rather than language model estimation.
    """
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=15
        )
        out = (result.stdout + "\n" + result.stderr).strip()
        return out if out else "Executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Error: Python code execution timed out (>15s)."
    except Exception as e:
        return f"Execution Error: {e}"


# ─────────────────────────────────────────────
# 🤖 GEMINI CONNECTOR
# Uses Google's Gemini 2.5 Flash with Function
# Calling enabled. Acts as the Orchestrator in
# Hive Auto Mode.
# ─────────────────────────────────────────────
class GeminiConnector:
    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            tools=[execute_python_code],
            system_instruction=(
                "You are NeuroSync's Orchestrator Node, powered by Gemini. "
                "You are the Manager of a multi-agent AI hive. "
                "For any task requiring computation or math, you MUST use "
                "your execute_python_code tool. Never estimate numerical results. "
                "Format all responses in clean, beautiful Markdown."
            )
        )

    def query(self, prompt: str) -> Tuple[str, str]:
        try:
            chat = self.model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(prompt)

            # Extract and display the orchestration steps
            orchestration_log = ""
            for msg in chat.history:
                for part in msg.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        try:
                            code = part.function_call.args['code']
                            orchestration_log += f"**⚡ Gemini wrote Python:**\n```python\n{code}\n```\n\n"
                        except Exception:
                            pass
                    elif hasattr(part, 'function_response') and part.function_response:
                        try:
                            res_str = str(dict(part.function_response.response))
                            if len(res_str) > 300:
                                res_str = res_str[:300] + "... (truncated)"
                            orchestration_log += f"**⚡ Executor returned:**\n```\n{res_str}\n```\n\n---\n\n"
                        except Exception:
                            pass

            return "Gemini-2.5-Flash", orchestration_log + response.text
        except Exception as e:
            return "System-Error", f"Gemini failed: {str(e)}"


# ─────────────────────────────────────────────
# 🤖 GPT-4o CONNECTOR (OpenAI)
# Uses OpenAI's GPT-4o. GPT-4o is best known
# for instruction following and code generation.
# In Hive Auto Mode, it is used for coding tasks.
# ─────────────────────────────────────────────
class OpenAIConnector:
    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

    def query(self, prompt: str) -> Tuple[str, str]:
        if not self.client:
            return "System-Error", "OpenAI API key not configured in .env"
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are NeuroSync's GPT-4o Node. You are a specialist in "
                            "writing clean, well-commented code and structured analysis. "
                            "Format all responses in clear, beautiful Markdown."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            return "GPT-4o", response.choices[0].message.content
        except Exception as e:
            return "System-Error", f"GPT-4o failed: {str(e)}"


# ─────────────────────────────────────────────
# 🤖 CLAUDE CONNECTOR (Anthropic)
# Uses Anthropic's Claude Sonnet. Claude is
# widely known as the best model for nuanced
# writing, analysis, and long-context reasoning.
# In Hive Auto Mode, it is used for writing tasks.
# ─────────────────────────────────────────────
class ClaudeConnector:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.api_key = api_key

    def query(self, prompt: str) -> Tuple[str, str]:
        if not self.api_key:
            return "System-Error", "Anthropic API key not configured in .env"
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                system=(
                    "You are NeuroSync's Claude Node. You are a specialist in nuanced "
                    "writing, long-form analysis, and creative problem solving. "
                    "Format all responses in clean, beautiful Markdown."
                ),
                messages=[{"role": "user", "content": prompt}]
            )
            return "Claude-Sonnet", response.content[0].text
        except Exception as e:
            return "System-Error", f"Claude failed: {str(e)}"


# ─────────────────────────────────────────────
# 🤖 LOCAL OLLAMA CONNECTOR
# Runs a local LLM (Llama 3.2) using Ollama.
# Completely private — no data leaves your machine.
# Falls back to Python Executor if Ollama is offline.
# ─────────────────────────────────────────────
class OllamaConnector:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"

    def query(self, prompt: str) -> Tuple[str, str]:
        import urllib.request, urllib.error
        try:
            data = json.dumps({
                "model": "llama3.2:latest",
                "prompt": prompt,
                "stream": False
            }).encode('utf-8')
            req = urllib.request.Request(
                self.ollama_url, data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return "Llama-3.2-Local", result.get("response", "No response")
        except Exception as e:
            print(f"Ollama offline: {e}. Falling back to Python Executor.")
            result = execute_python_code(prompt)
            return "Python-Executor-Fallback", result


# ─────────────────────────────────────────────
# ⚡ PYTHON EXECUTOR CONNECTOR
# Directly runs Python code. Used for math,
# data processing, and system-level tasks.
# ─────────────────────────────────────────────
class PythonExecutorConnector:
    def query(self, prompt: str) -> Tuple[str, str]:
        # Try to extract code from the prompt
        clean_expr = "".join(c for c in prompt if c in "0123456789+-*/().% ")
        if clean_expr.strip():
            try:
                result = eval(clean_expr.strip())
                return "Python-Executor", f"**Result:** `{result}`"
            except Exception:
                pass
        # Run the raw prompt as code
        result = execute_python_code(prompt)
        return "Python-Executor", f"```\n{result}\n```"


# ─────────────────────────────────────────────
# 🐝 HIVE ORCHESTRATOR (Auto Mode)
# The crown jewel. Analyzes your task and
# automatically routes it to the best specialist.
#
# KEY CONCEPT: This is "Agentic Routing" — the
# Orchestrator AI (Gemini) reads the task and
# outputs a routing decision in JSON. We parse
# that JSON and call the appropriate specialist.
# ─────────────────────────────────────────────
class HiveOrchestrator:
    def __init__(self):
        self.gemini = GeminiConnector()
        self.openai = OpenAIConnector()
        self.claude = ClaudeConnector()
        self.ollama = OllamaConnector()
        self.executor = PythonExecutorConnector()

    def query(self, prompt: str) -> Tuple[str, str]:
        # Step 1: Use Gemini as the routing brain to classify the task
        try:
            import google.generativeai as genai
            router_model = genai.GenerativeModel(
                'gemini-2.5-flash',
                system_instruction=(
                    "You are a task router for a multi-agent AI system. "
                    "Analyze the user's task and reply with ONLY a JSON object (no markdown, no explanation). "
                    "The JSON must have two keys: "
                    "'agent' (one of: 'gemini', 'openai', 'claude', 'ollama', 'executor') and "
                    "'reason' (a one-sentence explanation). "
                    "Rules: "
                    "- Tasks requiring math, computation, or running code → 'gemini' (has code executor tool). "
                    "- Tasks requiring code writing, debugging, technical explanation → 'openai'. "
                    "- Tasks requiring creative writing, analysis, summarization, essays → 'claude'. "
                    "- Tasks requiring private/offline processing → 'ollama'. "
                    "- Simple arithmetic only → 'executor'."
                )
            )
            routing_response = router_model.generate_content(prompt)
            raw = routing_response.text.strip().strip("```json").strip("```").strip()
            decision = json.loads(raw)
            agent = decision.get("agent", "gemini")
            reason = decision.get("reason", "Routed by Hive Orchestrator")
        except Exception as e:
            # If routing fails, fall back to Gemini
            agent = "gemini"
            reason = f"Routing failed ({e}), defaulted to Gemini."

        # Step 2: Dispatch to the chosen specialist
        routing_header = f"🐝 **Hive Routed → {agent.upper()}** _{reason}_\n\n---\n\n"
        if agent == "openai":
            node, text = self.openai.query(prompt)
        elif agent == "claude":
            node, text = self.claude.query(prompt)
        elif agent == "ollama":
            node, text = self.ollama.query(prompt)
        elif agent == "executor":
            node, text = self.executor.query(prompt)
        else:  # gemini (default + computation tasks)
            node, text = self.gemini.query(prompt)

        return f"Hive→{node}", routing_header + text
