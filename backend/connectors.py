import json
import os
import io
import sys
import warnings
from typing import Tuple
from dotenv import load_dotenv

# Suppress deprecated SDK warnings (Bug #5)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    Uses in-process exec() to avoid subprocess forks that crash gRPC.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = sys.stdout = io.StringIO()
    redirected_error = sys.stderr = io.StringIO()
    
    try:
        # Execute in a clean namespace
        exec(code, {"__builtins__": __builtins__})
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    out = (redirected_output.getvalue() + "\n" + redirected_error.getvalue()).strip()
    return out if out else "Executed successfully with no output."


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
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                return "System-Error", "⚠️ **Gemini 2.5 Flash is rate-limited** (free tier: 20 req/day). Try again tomorrow or switch to a local model."
            return "System-Error", f"⚠️ Gemini 2.5 Flash error: {err[:120]}"


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
            err = str(e)
            if "429" in err or "quota" in err.lower():
                return "System-Error", "⚠️ **Code Specialist is rate-limited.** Try again later or switch to Llama 3.2 (Local)."
            return "System-Error", f"⚠️ Code Specialist error: {err[:120]}"


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
            err = str(e)
            if "429" in err or "quota" in err.lower():
                return "System-Error", "⚠️ **Writing Specialist is rate-limited.** Try again later or switch to Llama 3.2 (Local)."
            return "System-Error", f"⚠️ Writing Specialist error: {err[:120]}"


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
            # Bug #2 Fix: Do not execute plain text as code if Ollama fails
            return "Ollama-Offline", "Error: Llama 3.2 is offline. Ensure Ollama is running on port 11434."


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
            routing_response = router_model.generate_content(
                prompt,
                request_options={"timeout": 8.0}
            )
            raw = routing_response.text.strip().strip("```json").strip("```").strip()
            decision = json.loads(raw)
            agent = decision.get("agent", "gemini")
            reason = decision.get("reason", "Routed by Hive Orchestrator")
        except Exception as e:
            # Bug #3 Fix: Instant offline fallback
            prompt_lower = prompt.lower()
            if any(w in prompt_lower for w in ["calculate", "math", "+", "-", "*", "/", "prime"]):
                agent = "gemini"
            elif any(w in prompt_lower for w in ["code", "debug", "python", "javascript", "function"]):
                agent = "coder"
            elif any(w in prompt_lower for w in ["write", "essay", "explain", "summarize"]):
                agent = "writer"
            elif any(w in prompt_lower for w in ["private", "local", "secret"]):
                agent = "ollama"
            else:
                agent = "gemini"
            reason = f"API offline ({str(e)[:50]}). Used instant keyword fallback."

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

        # Step 4: Call the specialist — cascade to Ollama if primary fails
        node_name, text = connector.query(prompt)
        
        # If the chosen cloud connector failed with a rate-limit, cascade to Ollama
        if node_name == "System-Error" and ("rate-limited" in text or "quota" in text.lower()):
            ollama_name, ollama_text = self.ollama.query(prompt)
            if ollama_name != "Ollama-Offline":
                cascade_header = routing_header + "\n> 🔄 _Gemini quota exhausted — cascaded to local Llama 3.2_\n\n---\n\n"
                return f"Hive→{ollama_name}", cascade_header + ollama_text
        
        return f"Hive→{node_name}", routing_header + text
