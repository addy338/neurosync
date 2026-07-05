"use client";

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useState, useRef, useEffect } from "react";

type Message = {
  id: number;
  text: string;
  sender: "user" | "ai";
  node?: string;
  streaming?: boolean; // true while the bubble is still receiving tokens
};

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("🐝 Auto Hive Mode");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 0,
      text: "NeuroSync initialized. Omni-Model routing active. Awaiting task.",
      sender: "ai",
      node: "System Core"
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom whenever messages change (including mid-stream updates)
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isTyping) return;

    const userText = prompt;
    const newUserMsg: Message = { id: Date.now(), text: userText, sender: "user" };
    setMessages(prev => [...prev, newUserMsg]);
    setPrompt("");
    setIsTyping(true);

    // Create a placeholder AI bubble that will be filled token by token
    const aiMsgId = Date.now() + 1;
    const placeholder: Message = {
      id: aiMsgId,
      text: "",
      sender: "ai",
      node: "Routing...",
      streaming: true,
    };
    setMessages(prev => [...prev, placeholder]);

    const controller = new AbortController();
    // 90s total timeout — long enough for Ollama on slow hardware
    const timeoutId = setTimeout(() => controller.abort(), 90000);

    try {
      // ─────────────────────────────────────────────
      // 📡 STREAMING FETCH
      // We call /tasks/stream/ which returns an SSE
      // stream. Instead of waiting for response.json(),
      // we read the body as a stream using a ReadableStream
      // reader and decode each chunk incrementally.
      //
      // 💡 LEARNING NOTE: fetch() returns a Response
      // whose .body is a ReadableStream<Uint8Array>.
      // We wrap it in a TextDecoder to get strings,
      // then split on "\n\n" to get SSE messages.
      // ─────────────────────────────────────────────
      const response = await fetch("http://localhost:8000/tasks/stream/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: userText, model }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode the binary chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // SSE messages are separated by "\n\n"
        const parts = buffer.split("\n\n");
        // The last element might be an incomplete message — keep it in the buffer
        buffer = parts.pop() ?? "";

        for (const part of parts) {
          if (!part.trim()) continue;

          // Parse the SSE event type and data
          const eventLine = part.match(/^event:\s*(.+)$/m);
          const dataLine  = part.match(/^data:\s*(.+)$/m);
          if (!eventLine || !dataLine) continue;

          const event = eventLine[1].trim();
          // The server escapes newlines as \n — restore them
          const data  = dataLine[1].replace(/\\n/g, "\n");

          if (event === "node") {
            // Routing decision header arrived — update the node label
            setMessages(prev => prev.map(msg =>
              msg.id === aiMsgId
                ? { ...msg, node: "Hive Orchestrator", text: data }
                : msg
            ));
          } else if (event === "chunk") {
            // Token arrived — append to the bubble text
            setMessages(prev => prev.map(msg =>
              msg.id === aiMsgId
                ? { ...msg, text: msg.text + data }
                : msg
            ));
          } else if (event === "done") {
            // Stream finished — data is the final node name
            setMessages(prev => prev.map(msg =>
              msg.id === aiMsgId
                ? { ...msg, node: data, streaming: false }
                : msg
            ));
          }
        }
      }
    } catch (error: any) {
      const msg = error?.name === "AbortError"
        ? "⚠️ Request timed out after 90 seconds. The backend may be overloaded or offline."
        : `⚠️ Could not reach the NeuroSync backend. Is FastAPI running on port 8000?\n\n_${error?.message ?? ""}_`;

      // Replace the placeholder bubble with the error
      setMessages(prev => prev.map(m =>
        m.id === aiMsgId
          ? { ...m, text: msg, node: "System Error", streaming: false }
          : m
      ));
    } finally {
      clearTimeout(timeoutId);
      setIsTyping(false);
    }
  };

  return (
    <main className="container">
      <header className="header">
        <h1>NeuroSync</h1>
      </header>

      <div className="glass-panel chat-box">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.sender}`}>
            {msg.sender === "ai" && (
              <div className="node-badge">
                <span style={{ fontSize: '1.2em' }}>⚡</span> {msg.node}
                {/* Blinking cursor while the bubble is streaming */}
                {msg.streaming && (
                  <span className="streaming-cursor" aria-hidden="true">▌</span>
                )}
              </div>
            )}
            <div className={`bubble ${msg.sender === "ai" ? "markdown-body" : ""}`}>
              {msg.sender === "ai" ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.text || "\u00a0" /* non-breaking space keeps bubble visible while empty */}
                </ReactMarkdown>
              ) : (
                msg.text
              )}
            </div>
          </div>
        ))}

        {/* Only show the "routing" spinner before the placeholder bubble appears */}
        {isTyping && messages[messages.length - 1]?.streaming && messages[messages.length - 1]?.text === "" && (
          <div className="message ai">
            <div className="node-badge">⚡ Orchestrator</div>
            <div className="bubble" style={{ fontStyle: 'italic', opacity: 0.7 }}>
              Routing to best specialist...
            </div>
          </div>
        )}

        <div ref={endOfMessagesRef} />
      </div>

      <form onSubmit={handleSubmit} className="input-area">
        <select
          className="prompt-input"
          style={{ width: 'auto', marginRight: '10px', padding: '0 10px' }}
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={isTyping}
        >
          <option value="🐝 Auto Hive Mode">🐝 Auto Hive Mode</option>
          <optgroup label="☁️ Gemini Family (Free)">
            <option value="Gemini 2.5 Flash (Cloud)">🧠 Gemini 2.5 Flash (Orchestrator)</option>
            <option value="Code Specialist (Gemini 2.0)">💻 Code Specialist (Gemini 2.0)</option>
            <option value="Writing Specialist (Gemini Lite)">✍️ Writing Specialist (Gemini Lite)</option>
          </optgroup>
          <optgroup label="🖥️ Local Models">
            <option value="Llama 3.2 (Local)">🦙 Llama 3.2 (Ollama)</option>
            <option value="Python Executor (Local)">⚡ Python Executor</option>
          </optgroup>
        </select>
        <input
          id="prompt-input"
          type="text"
          className="prompt-input"
          placeholder="Enter a complex task for the hive mind..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={isTyping}
        />
        <button type="submit" className="send-button" disabled={isTyping}>
          {isTyping ? "..." : "Sync"}
        </button>
      </form>
    </main>
  );
}
