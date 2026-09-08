import { useState, useLayoutEffect, useRef, useCallback, useEffect } from "preact/hooks";
import type { ComponentChild } from "preact";
import { fetchWithFallback } from "../../lib/ai-client";

interface OllamaModel {
  name: string;
  modified_at: string;
  size: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

interface DebugEntry {
  type: "request" | "response" | "stream_chunk" | "error" | "info";
  timestamp: number;
  data: unknown;
}

interface DebugStats {
  startTime: number | null;
  endTime: number | null;
  firstTokenTime: number | null;
  totalTokens: number;
  chunksReceived: number;
  requestSize: number;
  responseSize: number;
}

const PRIMARY_AI_ENDPOINT = import.meta.env.PUBLIC_PRIMARY_AI_ENDPOINT || "https://llm.laravelmail.com";
const FALLBACK_AI_ENDPOINT = import.meta.env.PUBLIC_FALLBACK_AI_ENDPOINT;

const presetAvatars = [
  "bg-gradient-to-br from-purple-500 to-pink-500",
  "bg-gradient-to-br from-cyan-500 to-blue-500",
  "bg-gradient-to-br from-green-400 to-emerald-500",
  "bg-gradient-to-br from-orange-400 to-red-500",
  "bg-gradient-to-br from-yellow-400 to-orange-500",
  "bg-gradient-to-br from-pink-400 to-rose-500",
  "bg-gradient-to-br from-indigo-400 to-purple-500",
  "bg-gradient-to-br from-teal-400 to-cyan-500",
];

const SUGGESTED_PROMPTS = [
  "Write an email to a colleague requesting a meeting to discuss a project",
  "Help me create a welcome email for new users",
  "Help me create a promotional email campaign",
  "Write a professional follow-up email after a job interview",
];

function getInitials(name: string) {
  return name.charAt(0).toUpperCase();
}

function formatTime(iso: string) {
  const d = new Date(iso);
  const now = new Date();
  const diff = now.getTime() - d.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  return `${Math.floor(hrs / 24)}d`;
}

function formatTimestamp(ts: number) {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function estimateTokens(text: string) {
  return Math.ceil(text.length / 4);
}

function renderMarkdown(text: string) {
  const parts: ComponentChild[] = [];
  const codeBlockRegex = /```(\w*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(renderInlineMarkdown(text.slice(lastIndex, match.index)));
    }
    const lang = match[1] || "text";
    const code = match[2].trimEnd();
    parts.push(
      <div key={match.index} className="relative group my-3 rounded-xl overflow-hidden border border-gray-200 dark:border-gray-600">
        <div className="flex items-center justify-between bg-gray-800 text-gray-300 text-xs px-4 py-2">
          <span className="font-mono">{lang}</span>
          <button
            onClick={() => {
              navigator.clipboard.writeText(code);
            }}
            className="flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-white text-[11px]"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
            Copy
          </button>
        </div>
        <pre className="bg-gray-900 text-gray-100 text-sm p-4 overflow-x-auto font-mono leading-relaxed">
          <code>{code}</code>
        </pre>
      </div>
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(renderBlockMarkdown(text.slice(lastIndex)));
  }

  return parts.length > 0 ? parts : renderInlineMarkdown(text);
}

function renderBlockMarkdown(text: string) {
  const blocks = text.split(/\n\n+/);
  return blocks.map((block, i) => {
    const trimmed = block.trim();
    if (!trimmed) return null;

    if (/^#{1,6}\s/.test(trimmed)) {
      const level = trimmed.match(/^(#{1,6})/)?.[1].length ?? 1;
      const content = trimmed.replace(/^#{1,6}\s+/, "");
      const sizeClasses: Record<number, string> = {
        1: "text-xl font-bold mt-4 mb-2",
        2: "text-lg font-bold mt-4 mb-2",
        3: "text-base font-semibold mt-3 mb-1",
        4: "text-sm font-semibold mt-3 mb-1",
        5: "text-sm font-medium mt-2 mb-1",
        6: "text-xs font-medium mt-2 mb-1",
      };
      const cls = sizeClasses[level] || sizeClasses[3];
      if (level === 1) return <h1 key={i} className={cls}>{renderInlineMarkdown(content)}</h1>;
      if (level === 2) return <h2 key={i} className={cls}>{renderInlineMarkdown(content)}</h2>;
      if (level === 3) return <h3 key={i} className={cls}>{renderInlineMarkdown(content)}</h3>;
      if (level === 4) return <h4 key={i} className={cls}>{renderInlineMarkdown(content)}</h4>;
      if (level === 5) return <h5 key={i} className={cls}>{renderInlineMarkdown(content)}</h5>;
      return <h6 key={i} className={cls}>{renderInlineMarkdown(content)}</h6>;
    }

    if (/^>\s/.test(trimmed)) {
      const content = trimmed.replace(/^>\s?/gm, "");
      return (
        <blockquote key={i} className="border-l-4 border-purple-400 pl-4 my-2 text-gray-600 dark:text-gray-300 italic">
          {renderInlineMarkdown(content)}
        </blockquote>
      );
    }

    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      return <hr key={i} className="my-3 border-gray-200 dark:border-gray-600" />;
    }

    if (/^[-*+]\s/.test(trimmed)) {
      const items = trimmed.split("\n").filter(l => /^[-*+]\s/.test(l.trim()));
      return (
        <ul key={i} className="list-disc list-inside space-y-1 my-2 text-gray-700 dark:text-gray-300">
          {items.map((item, j) => (
            <li key={j}>{renderInlineMarkdown(item.replace(/^[-*+]\s+/, ""))}</li>
          ))}
        </ul>
      );
    }

    if (/^\d+\.\s/.test(trimmed)) {
      const items = trimmed.split("\n").filter(l => /^\d+\.\s/.test(l.trim()));
      return (
        <ol key={i} className="list-decimal list-inside space-y-1 my-2 text-gray-700 dark:text-gray-300">
          {items.map((item, j) => (
            <li key={j}>{renderInlineMarkdown(item.replace(/^\d+\.\s+/, ""))}</li>
          ))}
        </ol>
      );
    }

    if (trimmed.includes("\n")) {
      return (
        <div key={i} className="space-y-1 my-1">
          {trimmed.split("\n").map((line, j) => (
            <p key={j} className="text-gray-700 dark:text-gray-300">{renderInlineMarkdown(line)}</p>
          ))}
        </div>
      );
    }

    return <p key={i} className="text-gray-700 dark:text-gray-300 my-1">{renderInlineMarkdown(trimmed)}</p>;
  });
}

function renderInlineMarkdown(text: string) {
  const parts: ComponentChild[] = [];
  const inlineRegex = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\))/g;
  let lastIdx = 0;
  let m;

  while ((m = inlineRegex.exec(text)) !== null) {
    if (m.index > lastIdx) {
      parts.push(text.slice(lastIdx, m.index));
    }
    if (m[0].startsWith("**")) {
      parts.push(<strong key={m.index} className="font-bold">{m[0].replace(/\*\*/g, "")}</strong>);
    } else if (m[0].startsWith("`")) {
      parts.push(
        <code key={m.index} className="bg-gray-200 dark:bg-gray-600 px-1.5 py-0.5 rounded text-sm font-mono text-pink-600 dark:text-pink-400">
          {m[0].replace(/`/g, "")}
        </code>
      );
    } else if (m[0].startsWith("[")) {
      const linkMatch = m[0].match(/\[([^\]]+)\]\(([^)]+)\)/);
      if (linkMatch) {
        parts.push(
          <a key={m.index} href={linkMatch[2]} target="_blank" rel="noopener noreferrer" className="text-purple-600 dark:text-purple-400 hover:underline">
            {linkMatch[1]}
          </a>
        );
      }
    }
    lastIdx = m.index + m[0].length;
  }

  if (lastIdx < text.length) {
    parts.push(text.slice(lastIdx));
  }

  return parts.length > 0 ? <>{parts}</> : text;
}

export default function AssistantChat() {
  const [models] = useState<OllamaModel[]>([
    { name: "hf.co/laravelcompany/laravelmail:latest", modified_at: new Date().toISOString(), size: 0 },
  ]);
  const [selectedModel] = useState<string>("hf.co/laravelcompany/laravelmail:latest");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [debugLogs, setDebugLogs] = useState<DebugEntry[]>([]);
  const [debugStats, setDebugStats] = useState<DebugStats>({
    startTime: null,
    endTime: null,
    firstTokenTime: null,
    totalTokens: 0,
    chunksReceived: 0,
    requestSize: 0,
    responseSize: 0,
  });
  const [copiedMessageIndex, setCopiedMessageIndex] = useState<number | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [dataProvider, setDataProvider] = useState<string>("local");
  const [dataModel, setDataModel] = useState<string>("hf.co/laravelcompany/laravelmail:latest");

  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const rightPanelRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const debugContainerRef = useRef<HTMLDivElement>(null);

  const addDebugLog = useCallback((entry: Omit<DebugEntry, "timestamp">) => {
    setDebugLogs((prev) => [...prev, { ...entry, timestamp: Date.now() }]);
  }, []);

  const scrollToBottom = () => {
    const el = messagesContainerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  };

  useLayoutEffect(() => {
    scrollToBottom();
  }, [messages]);

  useLayoutEffect(() => {
    const updateMessagesHeight = () => {
      if (!messagesContainerRef.current || !rightPanelRef.current) return;
      const rect = rightPanelRef.current.getBoundingClientRect();
      const header = rightPanelRef.current.querySelector<HTMLElement>('[data-part="header"]');
      const input = rightPanelRef.current.querySelector<HTMLElement>('[data-part="input"]');
      const headerH = header?.offsetHeight ?? 0;
      const inputH = input?.offsetHeight ?? 0;
      messagesContainerRef.current.style.maxHeight = `${rect.height - headerH - inputH}px`;
    };
    updateMessagesHeight();
    window.addEventListener("resize", updateMessagesHeight);
    return () => window.removeEventListener("resize", updateMessagesHeight);
  }, []);

  useEffect(() => {
    if (debugContainerRef.current) {
      debugContainerRef.current.scrollTop = debugContainerRef.current.scrollHeight;
    }
  }, [debugLogs]);

  const clearChat = () => {
    setMessages([]);
    setDebugLogs([]);
    setDebugStats({
      startTime: null,
      endTime: null,
      firstTokenTime: null,
      totalTokens: 0,
      chunksReceived: 0,
      requestSize: 0,
      responseSize: 0,
    });
    addDebugLog({ type: "info", data: "Chat cleared" });
  };

  const copyToClipboard = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedMessageIndex(index);
    setTimeout(() => setCopiedMessageIndex(null), 2000);
  };

  const sendMessage = async (promptText?: string) => {
    const messageText = promptText || input.trim();
    if (!messageText || isLoading || !selectedModel) return;

    const userMsg: Message = { role: "user", content: messageText, timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    const assistantMsg: Message = { role: "assistant", content: "", timestamp: Date.now() };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      const fullPrompt = promptText || input.trim();
      const apiBase = import.meta.env.SSR ? "" : import.meta.env.PUBLIC_API_BASE_URL || "";
      const apiUrl = apiBase + "/api/ai-generate?prompt=" + encodeURIComponent(fullPrompt);

      const response = await fetch(apiUrl, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "AI generation failed");
      }

      const data = await response.json();

      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === "assistant") {
          updated[updated.length - 1] = {
            ...last,
            content: data.data?.text || response.statusText,
          };
        }
        return updated;
      });

      // Update provider/model state for debug panel
      if (data.data) {
        setDataProvider(data.data.provider || "local");
        setDataModel(data.data.model || "hf.co/laravelcompany/laravelmail:latest");
      }

      // Update debug panel with provider info
      addDebugLog({
        type: "info",
        data: `Provider: ${data.data?.provider || "unknown"}, Model: ${data.data?.model || "unknown"}`,
      });

      addDebugLog({
        type: "info",
        data: `AI response received after ${Date.now() - (debugStats.startTime ?? Date.now())}ms`,
      });
    } catch (e) {
      console.error("Chat error:", e);
      addDebugLog({ type: "error", data: e instanceof Error ? e.message : String(e) });
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === "assistant") {
          updated[updated.length - 1] = {
            ...last,
            content: `Error: Unable to generate email. Please try again later.`,
          };
        }
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const elapsed = debugStats.startTime
    ? ((debugStats.endTime ?? Date.now()) - debugStats.startTime) / 1000
    : 0;

  return (
    <div style={{ height: "calc(100vh - 80px)" }} className="flex flex-col border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-gray-900">
      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar - model selector */}
        <div className={`${sidebarOpen ? "translate-x-0" : "-translate-x-full"} md:translate-x-0 fixed md:static inset-y-0 left-0 z-30 w-72 md:w-72 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 flex flex-col transition-transform duration-200 ease-in-out`}>
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Models</h2>
            <button onClick={() => setSidebarOpen(false)} className="md:hidden p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800">
              <svg className="w-5 h-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          </div>
          <div className="flex-1 overflow-y-auto">
            <ul className="divide-y divide-gray-100 dark:divide-gray-800">
              {models.map((model) => (
                <li key={model.name} className="p-3 bg-blue-50/50 dark:bg-gray-800/50 cursor-default">
                  <div className="flex items-center space-x-3">
                    <div className="relative flex-shrink-0">
                      <div className={`w-9 h-9 rounded-full flex items-center justify-center text-white font-bold text-xs ${presetAvatars[0]}`}>
                        {getInitials(model.name)}
                      </div>
                      <span className="absolute h-2.5 w-2.5 rounded-full border-2 border-white dark:border-gray-900 -bottom-0.5 -right-0.5 bg-green-400" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate dark:text-white">{model.name}</p>
                      <p className="text-xs text-gray-500 truncate dark:text-gray-400">Online</p>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Mobile sidebar overlay */}
        {sidebarOpen && (
          <div className="fixed inset-0 bg-black/50 z-20 md:hidden" onClick={() => setSidebarOpen(false)} />
        )}

        {/* Right panel */}
        <div ref={rightPanelRef} className="flex-1 flex flex-col bg-gray-50 dark:bg-gray-800 min-w-0">
          {/* Header */}
          <div data-part="header" className="flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
            <div className="flex items-center space-x-3 min-w-0">
              <button onClick={() => setSidebarOpen(true)} className="md:hidden p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800">
                <svg className="w-5 h-5 text-gray-600 dark:text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>
              </button>
              <div className="relative flex-shrink-0">
                <div className={`w-9 h-9 rounded-full flex items-center justify-center text-white font-bold text-xs ${presetAvatars[0]}`}>
                  {getInitials(selectedModel)}
                </div>
                <span className={`absolute h-2.5 w-2.5 rounded-full border-2 border-white dark:border-gray-900 -left-0.5 -top-0.5 ${isLoading ? "bg-yellow-400 animate-pulse" : "bg-green-400"}`} />
              </div>
              <div className="min-w-0">
                <h3 className="text-sm font-semibold text-gray-900 dark:text-white truncate">{selectedModel}</h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">{isLoading ? "Typing..." : "Ready"}</p>
              </div>
            </div>
            <div className="flex items-center space-x-1 flex-shrink-0">
              <button
                onClick={clearChat}
                disabled={messages.length === 0}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                title="Clear chat"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
              </button>
              <button
                onClick={() => setShowDebug(!showDebug)}
                className={`p-2 rounded-lg transition-colors ${showDebug ? "bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400" : "text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"}`}
                title="Toggle debug panel"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" /></svg>
              </button>
              <button
                onClick={() => {
                  const msg = prompt("Enter API base URL:", PRIMARY_AI_ENDPOINT);
                  if (msg && msg !== PRIMARY_AI_ENDPOINT) {
                    addDebugLog({ type: "info", data: `Note: URL change requires code edit. Current: ${PRIMARY_AI_ENDPOINT}` });
                  }
                }}
                className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                title="API settings"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
              </button>
            </div>
          </div>

          <div className="flex flex-1 overflow-hidden">
            {/* Messages area */}
            <div className="flex-1 flex flex-col min-w-0">
              <div ref={messagesContainerRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-4" style={{ minHeight: 0, WebkitOverflowScrolling: "touch" }}>
                {messages.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-full text-center px-4">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center mb-4 shadow-lg">
                      <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" /></svg>
                    </div>
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-1">Generate professional emails with AI</h2>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-6 max-w-sm">Describe the email you need — the free AI email writer will draft it for you in seconds.</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-lg w-full">
                      {SUGGESTED_PROMPTS.map((prompt) => (
                        <button
                          key={prompt}
                          onClick={() => sendMessage(prompt)}
                          className="text-left px-3 py-2.5 text-sm text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-purple-300 dark:hover:border-purple-500 transition-all duration-150 shadow-sm"
                        >
                          {prompt}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {messages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} group`}>
                    {msg.role === "assistant" && (
                      <div className="flex-shrink-0 mr-2 mt-1">
                        <div className={`w-7 h-7 rounded-full flex items-center justify-center text-white text-[10px] font-bold ${presetAvatars[0]}`}>
                          {getInitials(selectedModel)}
                        </div>
                      </div>
                    )}
                    <div className="max-w-[80%] sm:max-w-md">
                      <div
                        className={`px-3.5 py-2.5 rounded-2xl shadow-sm text-sm leading-relaxed ${
                          msg.role === "user"
                            ? "bg-purple-600 text-white rounded-br-md"
                            : "bg-white dark:bg-gray-700 text-gray-900 dark:text-white rounded-bl-md border border-gray-200 dark:border-gray-600"
                        }`}
                      >
                        {msg.role === "assistant" ? (
                          <div className="space-y-2">{renderMarkdown(msg.content)}</div>
                        ) : (
                          <p className="whitespace-pre-wrap">{msg.content}</p>
                        )}
                        {msg.role === "assistant" && isLoading && i === messages.length - 1 && !msg.content && (
                          <div className="flex space-x-1.5 py-1">
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                            <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                          </div>
                        )}
                      </div>
                      <div className={`flex items-center mt-1 space-x-2 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                        <span className="text-[10px] text-gray-400 dark:text-gray-500">{formatTimestamp(msg.timestamp)}</span>
                        {msg.content && (
                          <button
                            onClick={() => copyToClipboard(msg.content, i)}
                            className="opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                            title="Copy message"
                          >
                            {copiedMessageIndex === i ? (
                              <svg className="w-3 h-3 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                            ) : (
                              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                            )}
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Input area */}
              <div data-part="input" className="px-4 py-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
                <div className="flex items-center space-x-2">
                  <input
                    ref={inputRef}
                    className="flex-1 border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    type="text"
                    placeholder="Describe the email you want to write... e.g. a meeting request or sales follow-up"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    disabled={isLoading}
                  />
                  <button
                    onClick={() => sendMessage()}
                    disabled={isLoading || !input.trim() || !selectedModel}
                    className="flex-shrink-0 p-2.5 bg-purple-600 text-white rounded-xl hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                  >
                    {isLoading ? (
                      <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" /></svg>
                    ) : (
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Debug panel */}
            {showDebug && (
              <div className="w-80 lg:w-96 border-l border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 flex flex-col flex-shrink-0">
                <div className="px-3 py-2.5 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Debug</span>
                    <span className="px-1.5 py-0.5 text-[10px] font-medium bg-purple-100 dark:bg-purple-900/40 text-purple-600 dark:text-purple-400 rounded">LIVE</span>
                  </div>
                  <button
                    onClick={() => setDebugLogs([])}
                    className="text-[10px] text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                  >
                    Clear
                  </button>
                </div>

                {/* Stats bar */}
                <div className="px-3 py-2 border-b border-gray-100 dark:border-gray-800 grid grid-cols-2 gap-2 text-[11px]">
                  <div className="bg-gray-50 dark:bg-gray-800 rounded px-2 py-1.5">
                    <span className="text-gray-400 dark:text-gray-500">Elapsed</span>
                    <p className="font-mono font-medium text-gray-900 dark:text-white">{elapsed.toFixed(1)}s</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-800 rounded px-2 py-1.5">
                    <span className="text-gray-400 dark:text-gray-500">First token</span>
                    <p className="font-mono font-medium text-gray-900 dark:text-white">
                      {debugStats.firstTokenTime ? `${debugStats.firstTokenTime}ms` : "-"}
                    </p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-800 rounded px-2 py-1.5">
                    <span className="text-gray-400 dark:text-gray-500">~Tokens</span>
                    <p className="font-mono font-medium text-gray-900 dark:text-white">{debugStats.totalTokens || "-"}</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-800 rounded px-2 py-1.5">
                    <span className="text-gray-400 dark:text-gray-500">Chunks</span>
                    <p className="font-mono font-medium text-gray-900 dark:text-white">{debugStats.chunksReceived || "-"}</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-800 rounded px-2 py-1.5">
                    <span className="text-gray-400 dark:text-gray-500">Req size</span>
                    <p className="font-mono font-medium text-gray-900 dark:text-white">{debugStats.requestSize ? `${debugStats.requestSize}B` : "-"}</p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-800 rounded px-2 py-1.5">
                    <span className="text-gray-400 dark:text-gray-500">Res size</span>
                    <p className="font-mono font-medium text-gray-900 dark:text-white">{debugStats.responseSize ? `${debugStats.responseSize}B` : "-"}</p>
                  </div>
                </div>

                {/* Log entries */}
                <div ref={debugContainerRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5" style={{ minHeight: 0 }}>
                  {debugLogs.length === 0 && (
                    <p className="text-xs text-gray-400 dark:text-gray-500 text-center py-4">Send a message to see debug logs</p>
                  )}
                  {debugLogs.map((log, i) => (
                    <div
                      key={i}
                      className={`text-[11px] rounded-lg px-2.5 py-2 font-mono ${
                        log.type === "error"
                          ? "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800"
                          : log.type === "request"
                          ? "bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-800"
                          : log.type === "response"
                          ? "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-800"
                          : log.type === "stream_chunk"
                          ? "bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-400 border border-amber-200 dark:border-amber-800"
                          : "bg-gray-50 dark:bg-gray-800 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-gray-700"
                      }`}
                    >
                      <div className="flex items-start justify-between mb-1">
                        <span className="font-semibold uppercase text-[9px] tracking-wider">{log.type}</span>
                        <span className="text-[9px] opacity-60">{new Date(log.timestamp).toLocaleTimeString()}</span>
                      </div>
                      <pre className="whitespace-pre-wrap break-all text-[10px] leading-relaxed">
                        {typeof log.data === "string" ? log.data : JSON.stringify(log.data, null, 2)}
                      </pre>
                    </div>
                  ))}
                </div>

                {/* API endpoint */}
                <div className="px-3 py-2 border-t border-gray-200 dark:border-gray-700">
                  <p className="text-[10px] text-gray-400 dark:text-gray-500 font-mono truncate" title="Provider">
                    {dataProvider}
                  </p>
                  <p className="text-[10px] text-gray-400 dark:text-gray-500 font-mono truncate" title="Model">
                    {dataModel}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
