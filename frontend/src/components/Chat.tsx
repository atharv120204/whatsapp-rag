import { useEffect, useRef, useState } from "react";
import { streamChat, type ChatEvent, type ToolCall } from "../api";

interface Turn {
  role: "user" | "assistant";
  text: string;
  toolCalls: ToolCall[];
  error?: string;
}

const SUGGESTIONS = [
  "How many messages did each person send?",
  "Who starts conversations most often?",
  "What time of day is this group most active?",
  "Who replies the fastest?",
  "What did we decide about the trip?",
  "Show me the photos someone shared of food",
  "Which day had the most messages, and what were we talking about?",
  "Who asks the most questions?",
];

/** Minimal markdown: bold, inline code, and pipe tables. */
function renderMarkdown(text: string): string {
  const escape = (s: string) =>
    s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const lines = escape(text).split("\n");
  const out: string[] = [];
  let table: string[][] = [];

  const flushTable = () => {
    if (!table.length) return;
    const [header, ...body] = table;
    out.push(
      "<table><thead><tr>" +
        header.map((c) => `<th>${c}</th>`).join("") +
        "</tr></thead><tbody>" +
        body
          .map(
            (row) =>
              "<tr>" + row.map((c) => `<td>${c}</td>`).join("") + "</tr>",
          )
          .join("") +
        "</tbody></table>",
    );
    table = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("|") && trimmed.endsWith("|")) {
      const cells = trimmed.slice(1, -1).split("|").map((c) => c.trim());
      // The |---|---| separator row carries no data.
      if (!cells.every((c) => /^:?-{2,}:?$/.test(c))) table.push(cells);
      continue;
    }
    flushTable();
    if (!trimmed) {
      out.push("");
      continue;
    }
    out.push(trimmed);
  }
  flushTable();

  return out
    .join("\n")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br/>")
    .replace(/^(?!<table|<\/p>)/, "<p>")
    .concat("</p>")
    .replace(/<p><\/p>/g, "")
    .replace(/<p>(<table)/g, "$1")
    .replace(/(<\/table>)<\/p>/g, "$1");
}

function ToolTrace({ calls, live }: { calls: ToolCall[]; live?: string }) {
  if (!calls.length && !live) return null;
  const label = live
    ? `Running ${live}...`
    : `${calls.length} tool call${calls.length === 1 ? "" : "s"}`;

  return (
    <details className="trace" open={Boolean(live)}>
      <summary>{label}</summary>
      <div className="trace-body">
        {calls.map((call, i) => {
          const sql = (call.arguments as { query?: string }).query;
          const isSql = call.name === "run_sql" && sql;
          const result = call.result as { row_count?: number; result_count?: number; error?: string } | null;
          return (
            <div className="tool-call" key={i}>
              <span className="tool-name">{call.name}</span>{" "}
              <span className="muted">{call.duration_ms}ms</span>
              <pre className={isSql ? "sql" : undefined}>
                {isSql ? sql : JSON.stringify(call.arguments, null, 1)}
              </pre>
              {result?.error ? (
                <pre style={{ color: "var(--red)" }}>{result.error}</pre>
              ) : (
                <div className="muted" style={{ marginTop: 5 }}>
                  {result?.row_count !== undefined
                    ? `${result.row_count} rows`
                    : result?.result_count !== undefined
                      ? `${result.result_count} results`
                      : "ok"}
                </div>
              )}
            </div>
          );
        })}
        {live && <div className="tool-call muted">calling {live}...</div>}
      </div>
    </details>
  );
}

export default function Chat({
  archive,
  ready,
  blockedReason,
}: {
  archive: string | null;
  ready: boolean;
  blockedReason?: string;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [liveCalls, setLiveCalls] = useState<ToolCall[]>([]);
  const [liveTool, setLiveTool] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, liveCalls, liveTool]);

  async function send(question: string) {
    if (!question.trim() || busy) return;
    setInput("");
    setBusy(true);
    setLiveCalls([]);
    setLiveTool(null);

    const history = turns.map((t) => ({
      role: t.role === "user" ? "user" : "model",
      text: t.text,
    }));
    setTurns((prev) => [
      ...prev,
      { role: "user", text: question, toolCalls: [] },
    ]);

    const collected: ToolCall[] = [];
    try {
      await streamChat(question, history, archive, (event: ChatEvent) => {
        if (event.type === "tool_call") {
          setLiveTool(event.name);
        } else if (event.type === "tool_result") {
          const { type, ...call } = event;
          void type;
          collected.push(call as ToolCall);
          setLiveCalls([...collected]);
          setLiveTool(null);
        } else if (event.type === "answer") {
          setTurns((prev) => [
            ...prev,
            {
              role: "assistant",
              text: event.text,
              toolCalls: event.tool_calls?.length
                ? event.tool_calls
                : collected,
            },
          ]);
        } else if (event.type === "error") {
          setTurns((prev) => [
            ...prev,
            {
              role: "assistant",
              text: "",
              toolCalls: collected,
              error: event.message,
            },
          ]);
        }
      });
    } catch (err) {
      setTurns((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "",
          toolCalls: collected,
          error: err instanceof Error ? err.message : String(err),
        },
      ]);
    } finally {
      setBusy(false);
      setLiveCalls([]);
      setLiveTool(null);
    }
  }

  return (
    <>
      <div className="chat-scroll">
        <div className="chat-inner">
          {!ready && blockedReason && (
            <div className="error-box" style={{ marginBottom: 20 }}>
              {blockedReason}
            </div>
          )}

          {turns.length === 0 && (
            <div className="empty">
              <div style={{ fontSize: 15, marginBottom: 8 }}>
                Ask anything about the archive.
              </div>
              <div style={{ fontSize: 13 }}>
                Counting questions are answered with SQL, so the numbers are
                exact. Open the trace under any answer to see the query.
              </div>
            </div>
          )}

          {turns.map((turn, i) => (
            <div className={`msg ${turn.role}`} key={i}>
              <div className="role">{turn.role === "user" ? "You" : "Archive"}</div>
              {turn.role === "assistant" && <ToolTrace calls={turn.toolCalls} />}
              {turn.error ? (
                <div className="error-box">{turn.error}</div>
              ) : (
                <div
                  className="bubble"
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(turn.text) }}
                />
              )}
            </div>
          ))}

          {busy && (
            <div className="msg assistant">
              <div className="role">Archive</div>
              <ToolTrace calls={liveCalls} live={liveTool ?? undefined} />
              <div className="thinking">
                <span className="dot" />
                {liveTool ? `running ${liveTool}` : "thinking"}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="composer">
        {turns.length === 0 && (
          <div className="suggestions">
            {SUGGESTIONS.map((s) => (
              <button
                className="suggestion"
                key={s}
                onClick={() => send(s)}
                disabled={!ready || busy}
              >
                {s}
              </button>
            ))}
          </div>
        )}
        <div className="composer-inner">
          <input
            type="text"
            placeholder={
              ready ? "Ask about the chat..." : (blockedReason ?? "Not ready")
            }
            value={input}
            disabled={!ready || busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send(input)}
          />
          <button
            className="primary"
            onClick={() => send(input)}
            disabled={!ready || busy || !input.trim()}
          >
            Ask
          </button>
        </div>
      </div>
    </>
  );
}
