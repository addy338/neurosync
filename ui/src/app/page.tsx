"use client";

import { useState, useRef, useEffect } from "react";

type Message = {
  id: number;
  text: string;
  sender: "user" | "ai";
  node?: string;
};

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [model, setModel] = useState("Llama 3.2 (Local)");
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

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    // Add user message
    const newUserMsg: Message = { id: Date.now(), text: prompt, sender: "user" };
    setMessages(prev => [...prev, newUserMsg]);
    setPrompt("");
    setIsTyping(true);

    try {
      // Send real request to FastAPI Backend
      const response = await fetch("http://localhost:8000/tasks/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: newUserMsg.text, model: model }),
      });

      const data = await response.json();
      
      const newAiMsg: Message = {
        id: data.id || Date.now() + 1,
        text: data.response_text || "Task completed.",
        sender: "ai",
        node: data.assigned_node || "System"
      };
      
      setMessages(prev => [...prev, newAiMsg]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: "Error: Could not connect to the NeuroSync backend. Is the FastAPI server running on port 8000?",
        sender: "ai",
        node: "System Error"
      }]);
    } finally {
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
              </div>
            )}
            <div className="bubble">
              {msg.text}
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="message ai">
            <div className="node-badge">⚡ Orchestrator</div>
            <div className="bubble" style={{ fontStyle: 'italic', opacity: 0.7 }}>
              Analyzing task and routing...
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
        >
          <option value="Llama 3.2 (Local)">Llama 3.2 (Local)</option>
          <option value="Python Executor (Local)">Python Executor (Local)</option>
          <option value="Gemini 2.5 Pro (Cloud)">Gemini 2.5 Pro (Cloud)</option>
        </select>
        <input
          type="text"
          className="prompt-input"
          placeholder="Enter a complex task for the hive mind..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button type="submit" className="send-button" disabled={isTyping}>
          Sync
        </button>
      </form>
    </main>
  );
}
