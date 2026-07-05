import json
import os
import io
import re
import sys
import signal
import threading
import warnings
from typing import Generator, Tuple
from dotenv import load_dotenv

# Suppress deprecated SDK warnings (Bug #5)
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

# ─────────────────────────────────────────────
# 🔐 Singleton, thread-safe Gemini configuration
# (Bug #23: every connector previously called
# genai.configure() in __init__ — redundant global
# state mutation with a real race condition under
# FastAPI's threadpool-executed sync routes.)
# ─────────────────────────────────────────────
_genai_lock = threading.Lock()
_genai_configured = False


def _ensure_gemini_configured():
    global _genai_configured
    if _genai_configured:
        return
    with _genai_lock:
        if _genai_configured:
            return
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Create backend/.env with GEMINI_API_KEY=..."
            )
        genai.configure(api_key=api_key)
        _genai_configured = True


# ─────────────────────────────────────────────
# 🔧 SHARED LOCAL TOOL
# This function is given to the Orchestrator as
# a callable tool. It runs Python code on the
# LOCAL machine — something no cloud AI can do
# on its own. This is NeuroSync's superpower.
#
# (Bugs #20/#21: this previously ran with full
# __builtins__, no import restrictions, and no
# timeout. Hardened below to a lightweight sandbox
# — an allowlist of safe modules, blocked dangerous
# builtins, and an 8s wall-clock timeout. This is
# NOT equivalent to real process isolation; if this
# service is ever exposed beyond localhost, that's
# a hard prerequisite before doing so.)
# ─────────────────────────────────────────────
_ALLOWED_MODULES = {
    "math", "statistics", "datetime", "json", "re", "itertools",
    "collections", "random", "decimal", "functools", "string",
}

_BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "help",
    "exit", "quit", "globals", "locals", "vars", "dir", "breakpoint",
}


def _safe_import(name, globals_=None, locals_=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED_MODULES:
        raise ImportError(
            f"Import of '{name}' is blocked in the sandboxed executor. "
            f"Allowed modules: {sorted(_ALLOWED_MODULES)}"
        )
    return __import__(name, globals_, locals_, fromlist, level)


def _build_safe_globals():
    safe_builtins = {
        k: v for k, v in __builtins__.items() if k not in _BLOCKED_NAMES
    } if isinstance(__builtins__, dict) else {
        k: getattr(__builtins__, k) for k in dir(__builtins__) if k not in _BLOCKED_NAMES
    }
    safe_builtins["__import__"] = _safe_import
    return {"__builtins__": safe_builtins}


_exec_lock = threading.Lock()  # Bug #24: serialize stdout/stderr redirect


class _ExecTimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _ExecTimeoutError("Execution exceeded time limit (8s)")


def execute_python_code(code: str, timeout_seconds: int = 8) -> str:
    """
    Executes code in a restricted namespace (no os/sys/subprocess imports,
    no open/exec/eval), with a hard wall-clock timeout, and with stdout/
    stderr capture serialized via a lock so concurrent requests can't
    interleave each other's output.
    """
    with _exec_lock:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        redirected_output = sys.stdout = io.StringIO()
        redirected_error = sys.stderr = io.StringIO()

        use_alarm = hasattr(signal, "SIGALRM")
        if use_alarm:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout_seconds)

        try:
            exec(code, _build_safe_globals())
        except _ExecTimeoutError as e:
            print(f"Execution Error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Execution Error: {e}", file=sys.stderr)
        finally:
            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            sys.stdout, sys.stderr = old_stdout, old_stderr

        out = (redirected_output.getvalue() + "\n" + redirected_error.getvalue()).strip()
        return out if out else "Executed successfully with no output."


def _extract_json_block(raw: str) -> str:
    """
    Bug #22: the original used raw.strip("```json").strip("```") which
    strips a *character set*, not the literal fence substring, and can
    silently mangle legitimate JSON. This uses regex to remove actual
    ```json ... ``` or ``` ... ``` fences.
    """
    raw = raw.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return raw


# ─────────────────────────────────────────────
# 📡 SSE HELPERS
# Server-Sent Events format: each chunk is a
# "data: ...\n\n" line. The frontend reads these
# with a ReadableStream consumer and appends text
# to the message bubble in real-time.
#
# 💡 LEARNING NOTE: SSE is simpler than WebSockets
# for one-directional server→client push. The client
# opens one long-lived HTTP connection and the server
# drips data down it. Perfect for streaming AI text.
# ─────────────────────────────────────────────
def _sse(event: str, data: str) -> str:
    """Format a single SSE message. Newlines inside data are escaped."""
    safe_data = data.replace("\n", "\\n")
    return f"event: {event}\ndata: {safe_data}\n\n"


# ─────────────────────────────────────────────
# 🤖 GEMINI CONNECTOR
# Uses Google's Gemini 2.5 Flash with Function
# Calling enabled. Acts as the Orchestrator in
# Hive Auto Mode.
# ─────────────────────────────────────────────
class GeminiConnector:
    def __init__(self):
        _ensure_gemini_configured()
        import google.generativeai as genai
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

    def query(self, prompt: str, original_prompt: str = None) -> Tuple[str, str]:
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

            # Fix 1d: response.text raises if the model ended on a function_call
            # instead of a text part (hits the SDK's internal turn cap). Catch it
            # and surface the orchestration log rather than a raw SDK traceback.
            try:
                final_text = response.text
            except Exception:
                final_text = (
                    "The model ran a tool but didn't produce a final text answer. "
                    "Here's what it attempted:\n\n" + orchestration_log
                    if orchestration_log else
                    "The model didn't return a text response. Try rephrasing your request."
                )
                return "Gemini-2.5-Flash", final_text

            return "Gemini-2.5-Flash", orchestration_log + final_text
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                return "System-Error", "⚠️ **Gemini 2.5 Flash is rate-limited** (free tier: 20 req/day). Try again tomorrow or switch to a local model."
            return "System-Error", f"⚠️ Gemini 2.5 Flash error: {err[:120]}"

    def stream_query(self, prompt: str, original_prompt: str = None) -> Generator[str, None, Tuple[str, str]]:
        """
        Yields SSE chunks as Gemini streams its response token by token.
        Function calls (tool use) can't be streamed mid-flight, so if
        Gemini decides to call execute_python_code, we collect the full
        response first and emit it as one chunk after the tool runs.

        💡 LEARNING NOTE: The Gemini SDK supports stream=True on
        generate_content(), but NOT on start_chat() with automatic
        function calling — the SDK needs the full response to know
        whether to invoke a tool. So streaming + tool use is a two-phase
        process: we stream the non-tool parts when possible.
        """
        try:
            # Use streaming for pure text responses (no tool calls expected)
            response_stream = self.model.generate_content(prompt, stream=True)
            full_text = ""
            for chunk in response_stream:
                if chunk.text:
                    full_text += chunk.text
                    yield _sse("chunk", chunk.text)
            return "Gemini-2.5-Flash", full_text
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                msg = "⚠️ **Gemini 2.5 Flash is rate-limited** (free tier: 20 req/day). Try again tomorrow or switch to a local model."
            else:
                msg = f"⚠️ Gemini 2.5 Flash error: {err[:120]}"
            yield _sse("chunk", msg)
            return "System-Error", msg


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
        _ensure_gemini_configured()
        import google.generativeai as genai
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

    def query(self, prompt: str, original_prompt: str = None) -> Tuple[str, str]:
        try:
            response = self.model.generate_content(prompt)
            return "Code-Specialist (Gemini-2.0-Flash)", response.text
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                return "System-Error", "⚠️ **Code Specialist is rate-limited.** Try again later or switch to Llama 3.2 (Local)."
            return "System-Error", f"⚠️ Code Specialist error: {err[:120]}"

    def stream_query(self, prompt: str, original_prompt: str = None) -> Generator[str, None, Tuple[str, str]]:
        try:
            response_stream = self.model.generate_content(prompt, stream=True)
            full_text = ""
            for chunk in response_stream:
                if chunk.text:
                    full_text += chunk.text
                    yield _sse("chunk", chunk.text)
            return "Code-Specialist (Gemini-2.0-Flash)", full_text
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                msg = "⚠️ **Code Specialist is rate-limited.** Try again later or switch to Llama 3.2 (Local)."
            else:
                msg = f"⚠️ Code Specialist error: {err[:120]}"
            yield _sse("chunk", msg)
            return "System-Error", msg


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
        _ensure_gemini_configured()
        import google.generativeai as genai
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

    def query(self, prompt: str, original_prompt: str = None) -> Tuple[str, str]:
        try:
            response = self.model.generate_content(prompt)
            return "Writing-Specialist (Gemini-2.5-Flash-Lite)", response.text
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                return "System-Error", "⚠️ **Writing Specialist is rate-limited.** Try again later or switch to Llama 3.2 (Local)."
            return "System-Error", f"⚠️ Writing Specialist error: {err[:120]}"

    def stream_query(self, prompt: str, original_prompt: str = None) -> Generator[str, None, Tuple[str, str]]:
        try:
            response_stream = self.model.generate_content(prompt, stream=True)
            full_text = ""
            for chunk in response_stream:
                if chunk.text:
                    full_text += chunk.text
                    yield _sse("chunk", chunk.text)
            return "Writing-Specialist (Gemini-2.5-Flash-Lite)", full_text
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                msg = "⚠️ **Writing Specialist is rate-limited.** Try again later or switch to Llama 3.2 (Local)."
            else:
                msg = f"⚠️ Writing Specialist error: {err[:120]}"
            yield _sse("chunk", msg)
            return "System-Error", msg


# ─────────────────────────────────────────────
# 🤖 LOCAL OLLAMA CONNECTOR
# Runs a local LLM (Llama 3.2) using Ollama.
# Completely private — no data leaves your machine.
#
# 💡 LEARNING NOTE: Ollama's API supports native
# streaming via "stream": true. Instead of waiting
# for the entire response, it sends newline-delimited
# JSON objects — one per generated token. We parse
# each line and forward it immediately to the browser.
# This is called NDJSON (Newline-Delimited JSON).
# ─────────────────────────────────────────────
class OllamaConnector:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"

    def query(self, prompt: str, original_prompt: str = None) -> Tuple[str, str]:
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
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return "Llama-3.2-Local", result.get("response", "No response")
        except Exception:
            return "Ollama-Offline", "Error: Llama 3.2 is offline. Ensure Ollama is running on port 11434."

    def stream_query(self, prompt: str, original_prompt: str = None) -> Generator[str, None, Tuple[str, str]]:
        """
        Uses Ollama's native streaming mode (stream: true).
        Ollama sends a newline-delimited JSON stream where each line is:
          {"model": "...", "response": "<token>", "done": false}
        We forward each token to the browser immediately via SSE.
        """
        import urllib.request, urllib.error
        try:
            data = json.dumps({
                "model": "llama3.2:latest",
                "prompt": prompt,
                "stream": True
            }).encode('utf-8')
            req = urllib.request.Request(
                self.ollama_url, data=data,
                headers={'Content-Type': 'application/json'}
            )
            full_text = ""
            with urllib.request.urlopen(req, timeout=60) as resp:
                for raw_line in resp:
                    line = raw_line.decode('utf-8').strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        token = obj.get("response", "")
                        if token:
                            full_text += token
                            yield _sse("chunk", token)
                        if obj.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
            return "Llama-3.2-Local", full_text
        except Exception:
            msg = "Error: Llama 3.2 is offline. Ensure Ollama is running on port 11434."
            yield _sse("chunk", msg)
            return "Ollama-Offline", msg


# ─────────────────────────────────────────────
# ⚡ PYTHON EXECUTOR CONNECTOR
# Directly evaluates arithmetic expressions.
# Safe: runs with empty builtins, regex-guarded.
# ─────────────────────────────────────────────
class PythonExecutorConnector:
    """
    Bug #20 (Critical): this previously fell through to
    execute_python_code(prompt) for ANY non-arithmetic text — meaning
    selecting this model and typing anything executed it as raw Python
    with full os/subprocess access. Now ONLY evaluates clean arithmetic
    expressions in a builtins-free eval; anything else is rejected rather
    than executed.
    """
    _SAFE_EXPR_RE = re.compile(r"^[0-9+\-*/().%\s]+$")

    def query(self, prompt: str, original_prompt: str = None) -> Tuple[str, str]:
        clean_expr = "".join(c for c in prompt if c in "0123456789+-*/().% ")
        if clean_expr.strip() and self._SAFE_EXPR_RE.match(clean_expr.strip()):
            try:
                result = eval(clean_expr.strip(), {"__builtins__": {}}, {})
                return "Python-Executor", f"**Result:** `{result}`"
            except Exception as e:
                return "Python-Executor", f"Could not evaluate expression: {e}"
        return (
            "Python-Executor",
            "The Python Executor node only evaluates arithmetic expressions "
            "(e.g. `2 * (3 + 4)`). For general code, use the Code Specialist "
            "or Auto Hive Mode."
        )

    def stream_query(self, prompt: str, original_prompt: str = None) -> Generator[str, None, Tuple[str, str]]:
        # Execution is instant — no real streaming needed. Emit as a single chunk.
        node_name, result = self.query(prompt)
        yield _sse("chunk", result)
        return node_name, result


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
        self.gemini = GeminiConnector()            # 🧠 Orchestrator + computation
        self.coder = CodeSpecialistConnector()     # 💻 Code generation (Gemini 2.0 Flash)
        self.writer = WritingSpecialistConnector() # ✍️ Writing & analysis (Gemini 2.5 Lite)
        self.ollama = OllamaConnector()            # 🦙 100% local, private
        self.executor = PythonExecutorConnector()  # ⚡ Arithmetic only

    # ─────────────────────────────────────────
    # 🗺️ SHARED ROUTING HELPER
    # Both query() and stream_query() need to decide
    # which specialist to call. Extracting this into
    # _route() means the logic lives in exactly one
    # place — no drift between the two code paths.
    # ─────────────────────────────────────────
    def _route(self, prompt: str) -> Tuple[str, str, object]:
        """
        Returns (agent_key, routing_header, connector).
        Tries the Gemini Flash Lite router first; falls back instantly
        to keyword matching if the API is unavailable or rate-limited.
        """
        _ensure_gemini_configured()
        import google.generativeai as genai

        agent_map = {
            "gemini":   ("🧠 Gemini 2.5 Flash (Orchestrator)", self.gemini),
            "coder":    ("💻 Code Specialist (Gemini 2.0 Flash)", self.coder),
            "writer":   ("✍️ Writing Specialist (Gemini 2.5 Lite)", self.writer),
            "ollama":   ("🦙 Llama 3.2 Local", self.ollama),
            "executor": ("⚡ Python Executor", self.executor),
        }

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
            raw = _extract_json_block(routing_response.text)
            decision = json.loads(raw)
            agent = decision.get("agent", "gemini")
            reason = decision.get("reason", "Routed by Hive Orchestrator")
        except Exception as e:
            # Instant offline keyword fallback — zero API calls needed
            prompt_lower = prompt.lower()
            if any(w in prompt_lower for w in ["calculate", "math", "+", "-", "*", "/", "prime"]):
                agent = "gemini"
            elif any(w in prompt_lower for w in ["code", "debug", "python", "javascript", "function"]):
                agent = "coder"
            elif any(w in prompt_lower for w in ["write", "essay", "explain", "summarize"]):
                agent = "writer"
            elif any(w in prompt_lower for w in [
                # Fix 1c: Widen Ollama routing — previously only matched "local",
                # "private", "secret". "use a different model", "use ollama",
                # "use llama" etc. all fell through to gemini (the very thing
                # the user was trying to avoid).
                "private", "local", "secret", "offline",
                "other model", "different model", "switch model", "another model",
                "ollama", "llama",
            ]):
                agent = "ollama"
            else:
                agent = "gemini"
            reason = f"API offline ({str(e)[:50]}). Used instant keyword fallback."

        label, connector = agent_map.get(agent, ("🧠 Gemini 2.5 Flash", self.gemini))
        routing_header = (
            f"🐝 **Hive Routed → {label}**\n"
            f"> _{reason}_\n\n---\n\n"
        )
        return agent, routing_header, connector

    def query(self, prompt: str, original_prompt: str = None) -> Tuple[str, str]:
        # Fix 1b: clean_prompt is the user's raw text, without any memory context
        # prepended. The specialist still gets the full enriched prompt (better answers),
        # but the cascade fallback gets the clean version so a small local model
        # isn't handed a wall of old error text instead of the actual question.
        clean_prompt = original_prompt or prompt
        agent, routing_header, connector = self._route(clean_prompt)

        # Specialist gets the full enriched prompt for best quality
        node_name, text = connector.query(prompt)

        if node_name == "System-Error" and ("rate-limited" in text or "quota" in text.lower()):
            # Cascade uses the CLEAN prompt — not the memory-enriched one
            ollama_name, ollama_text = self.ollama.query(clean_prompt)
            if ollama_name != "Ollama-Offline":
                cascade_header = routing_header + "\n> 🔄 _Gemini quota exhausted — cascaded to local Llama 3.2_\n\n---\n\n"
                return f"Hive→{ollama_name}", cascade_header + ollama_text

        return f"Hive→{node_name}", routing_header + text

    def stream_query(self, prompt: str, original_prompt: str = None) -> Generator[str, None, Tuple[str, str]]:
        """
        Streaming version of query(). Emits the routing header as the
        first SSE chunk so the browser can show which node was chosen
        before the actual response starts arriving.
        """
        clean_prompt = original_prompt or prompt
        agent, routing_header, connector = self._route(clean_prompt)

        # Emit the routing decision immediately — no waiting for the model
        yield _sse("node", routing_header)

        # Stream from the chosen specialist (full enriched prompt for quality)
        gen = connector.stream_query(prompt)
        full_text = ""
        try:
            while True:
                chunk_sse = next(gen)
                # Extract text from SSE to build accumulated full_text for DB
                if chunk_sse.startswith("event: chunk\n"):
                    line = chunk_sse.split("data: ", 1)[1].rstrip("\n")
                    full_text += line.replace("\\n", "\n")
                yield chunk_sse
        except StopIteration as e:
            node_name_from_gen, _ = e.value if e.value else ("Unknown", "")
            node_name = node_name_from_gen

        # Cascade to Ollama if the primary model was rate-limited.
        # Use clean_prompt so Ollama gets the user's actual question, not
        # a prompt stuffed with old error text from memory recall.
        if "rate-limited" in full_text or "quota" in full_text.lower():
            cascade_note = "\n\n> 🔄 _Gemini quota exhausted — cascading to local Llama 3.2..._\n\n"
            yield _sse("chunk", cascade_note)
            full_text = cascade_note

            ollama_gen = self.ollama.stream_query(clean_prompt)
            try:
                while True:
                    chunk_sse = next(ollama_gen)
                    if chunk_sse.startswith("event: chunk\n"):
                        line = chunk_sse.split("data: ", 1)[1].rstrip("\n")
                        full_text += line.replace("\\n", "\n")
                    yield chunk_sse
            except StopIteration as e:
                node_name_from_ollama, _ = e.value if e.value else ("Ollama", "")
                node_name = f"Hive→{node_name_from_ollama}"
        else:
            node_name = f"Hive→{node_name}"

        # Final SSE event carries the node name so the frontend can label the bubble
        yield _sse("done", node_name)
        return node_name, routing_header + full_text
