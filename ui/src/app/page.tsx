"use client";

import { useState } from "react";

type Message = {
  id: number;
  text: string;
  sender: "user" | "ai";
  node?: string;
};

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "NeuroSync initialized. Connected to Local and Cloud AI nodes. Awaiting complex task routing.",
      sender: "ai",
      node: "System Core"
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    // Add user message
    const newUserMsg: Message = { id: Date.now(), text: prompt, sender: "user" };
    setMessages(prev => [...prev, newUserMsg]);
    setPrompt("");
    setIsTyping(true);

    // Simulate backend routing and response
    setTimeout(() => {
      const nodes = ["Claude-3.5-Sonnet", "Llama-3-Local", "Open-Interpreter", "Gemini-1.5-Pro"];
      const randomNode = nodes[Math.floor(Math.random() * nodes.length)];
      
      const newAiMsg: Message = {
        id: Date.now() + 1,
        text: `Task delegated successfully. Processed output from requested operation.`,
        sender: "ai",
        node: randomNode
      };
      
      setMessages(prev => [...prev, newAiMsg]);
      setIsTyping(false);
    }, 1500);
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
      </div>

      <form onSubmit={handleSubmit} className="input-area">
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
