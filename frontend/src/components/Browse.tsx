import { useEffect, useState } from "react";
import { api, type MessageRow } from "../api";

const PAGE = 100;

export default function Browse({ archive }: { archive: string | null }) {
  const [rows, setRows] = useState<MessageRow[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [query, setQuery] = useState("");
  const [sender, setSender] = useState("");
  const [people, setPeople] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.participants(archive).then((p) => setPeople(p.map((x) => x.display_name)));
  }, [archive]);

  useEffect(() => {
    setLoading(true);
    api
      .messages(archive, { q: query, sender, limit: PAGE, offset })
      .then((r) => {
        setRows(r.messages);
        setTotal(r.total);
      })
      .finally(() => setLoading(false));
  }, [archive, query, sender, offset]);

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="card">
        <h3>Browse the archive</h3>
        <div className="sub">
          Media rows show the AI-generated description inline, so search covers
          them too.
        </div>
        <div className="row">
          <input
            type="search"
            placeholder="Search message text..."
            value={query}
            style={{ flex: 1, minWidth: 240 }}
            onChange={(e) => {
              setOffset(0);
              setQuery(e.target.value);
            }}
          />
          <select
            value={sender}
            onChange={(e) => {
              setOffset(0);
              setSender(e.target.value);
            }}
          >
            <option value="">Everyone</option>
            {people.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
        <div className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>
          {total.toLocaleString()} matching messages
        </div>
      </div>

      <div className="card">
        {loading ? (
          <div className="empty">Loading...</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th style={{ width: 140 }}>When</th>
                <th style={{ width: 130 }}>Who</th>
                <th>Message</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.msg_id}>
                  <td className="muted" style={{ fontSize: 12 }}>
                    {row.ts.slice(0, 16).replace("T", " ")}
                  </td>
                  <td>{row.sender}</td>
                  <td>
                    {row.text}
                    {row.type !== "text" && (
                      <span className="muted" style={{ fontSize: 11.5 }}>
                        {" "}
                        [{row.type}]
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="row" style={{ marginTop: 14 }}>
          <button
            className="ghost"
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - PAGE))}
          >
            Previous
          </button>
          <span className="muted" style={{ fontSize: 13 }}>
            {offset + 1}–{Math.min(offset + PAGE, total)} of {total.toLocaleString()}
          </span>
          <button
            className="ghost"
            disabled={offset + PAGE >= total}
            onClick={() => setOffset(offset + PAGE)}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
