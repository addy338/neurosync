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
# 🤖 CODE SPECIALIST (Gemini 2.0 Flash)
# Gemini 2.0 Flash is optimized for speed and
# code generation tasks. In Hive Auto Mode, this
# node receives any coding/debugging tasks.
#
# 💡 LEARNING NOTE: This is called "Specialization".
# Even within the same model family, different
# versions have different speed/capability tradeoffs.
# gemini-2.0-flash prioritizes low-latency responses,
# making it ideal for iterative coding tasks.
# ─────────────────────────────────────────────
class CodeSpecialistConnector:
    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-2.0-flash',
            system_instruction=(
                "You are NeuroSync's Code Specialist Node, powered by Gemini 2.0 Flash. "
                "You are a world-class software engineer. "
                "Write clean, well-commented, production-quality code. "
                "Always explain what your code does step by step. "
                "Format all responses in beautiful Markdown with proper code blocks."
            )
        )

    def query(self, prompt: str) -> Tuple[str, str]:
        try:
            response = self.model.generate_content(prompt)
            return "Code-Specialist (Gemini-2.0-Flash)", response.text
        except Exception as e:
            return "System-Error", f"Code Specialist failed: {str(e)}"


# ─────────────────────────────────────────────
# ✍️ WRITING SPECIALIST (Gemini 2.5 Flash Lite)
# Gemini 2.5 Flash Lite is the most lightweight,
# cost-efficient model in the family. It is ideal
# for high-throughput writing and summarization.
#
# 💡 LEARNING NOTE: "Lite" models exist because
# not every task needs a massive reasoning model.
# Asking a billion-parameter model to write a short
# paragraph is wasteful. This is "right-sizing"
# your AI — a key principle in production systems.
# ─────────────────────────────────────────────
class WritingSpecialistConnector:
    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash-lite',
            system_instruction=(
                "You are NeuroSync's Writing Specialist Node, powered by Gemini 2.5 Flash Lite. "
                "You are an expert writer, analyst, and communicator. "
                "Produce nuanced, well-structured, deeply insightful content. "
                "Use clear headings, bullet points, and examples. "
                "Format all responses in beautiful Markdown."
            )
        )

    def query(self, prompt: str) -> Tuple[str, str]:
        try:
            response = self.model.generate_content(prompt)
            return "Writing-Specialist (Gemini-2.5-Flash-Lite)", response.text
        except Exception as e:
            return "System-Error", f"Writing Specialist failed: {str(e)}"


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
# Orchestrator AI (Gemini 2.5 Flash) reads the task
# and outputs a routing decision in JSON. We parse
# that JSON and call the appropriate specialist.
#
# 💡 LEARNING NOTE: This pattern is called the
# "Manager-Worker" or "Orchestrator-Subagent" pattern.
# It is the foundation of frameworks like LangGraph,
# AutoGen, and CrewAI. You are building it from scratch.
# ─────────────────────────────────────────────
class HiveOrchestrator:
    def __init__(self):
        self.gemini = GeminiConnector()          # 🧠 Orchestrator + computation
        self.coder = CodeSpecialistConnector()   # 💻 Code generation (Gemini 2.0 Flash)
        self.writer = WritingSpecialistConnector() # ✍️ Writing & analysis (Gemini 2.5 Lite)
        self.ollama = OllamaConnector()          # 🦙 100% local, private
        self.executor = PythonExecutorConnector() # ⚡ Raw Python execution

    def query(self, prompt: str) -> Tuple[str, str]:
        import google.generativeai as genai
        # Step 1: Use a lightweight Gemini model as the routing brain
        # It only has ONE job: classify the task type and output a JSON decision.
        # It does NOT answer the question itself.
        try:
            router_model = genai.GenerativeModel(
                'gemini-2.0-flash-lite',
                system_instruction=(
                    "You are a task router for a multi-agent AI system. "
                    "Analyze the user's task and reply with ONLY a JSON object (no markdown, no explanation). "
                    "The JSON must have two keys: "
                    "'agent' (one of: 'gemini', 'coder', 'writer', 'ollama', 'executor') and "
                    "'reason' (a one-sentence explanation). "
                    "Routing Rules: "
                    "- Math, computation, data analysis, running code → 'gemini' (has Python executor tool). "
                    "- Writing code, debugging, explaining technical concepts → 'coder'. "
                    "- Essays, analysis, summaries, creative writing, explanations → 'writer'. "
                    "- Tasks requiring privacy/offline processing → 'ollama'. "
                    "- Simple arithmetic only → 'executor'."
                )
            )
            routing_response = router_model.generate_content(prompt)
            raw = routing_response.text.strip().strip("```json").strip("```").strip()
            decision = json.loads(raw)
            agent = decision.get("agent", "gemini")
            reason = decision.get("reason", "Routed by Hive Orchestrator")
        except Exception as e:
            agent = "gemini"
            reason = f"Routing failed ({e}), defaulted to Gemini Orchestrator."

        # Step 2: Map the routing decision to the actual connector
        agent_map = {
            "gemini": ("🧠 Gemini 2.5 Flash (Orchestrator)", self.gemini),
            "coder": ("💻 Code Specialist (Gemini 2.0 Flash)", self.coder),
            "writer": ("✍️ Writing Specialist (Gemini 2.5 Lite)", self.writer),
            "ollama": ("🦙 Llama 3.2 Local", self.ollama),
            "executor": ("⚡ Python Executor", self.executor),
        }
        label, connector = agent_map.get(agent, ("🧠 Gemini 2.5 Flash", self.gemini))

        # Step 3: Build a header that shows the routing decision in the UI
        routing_header = (
            f"🐝 **Hive Routed → {label}**\n"
            f"> _{reason}_\n\n---\n\n"
        )

        # Step 4: Call the specialist
        node_name, text = connector.query(prompt)
        return f"Hive→{node_name}", routing_header + text
