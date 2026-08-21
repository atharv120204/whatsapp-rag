import { useEffect, useState } from "react";
import { api } from "../api";

const MOMENT_TABS: { id: string; label: string; blurb: string }[] = [
  { id: "funny", label: "Funniest", blurb: "Far more laughter than this chat's normal level." },
  { id: "argument", label: "Friction", blurb: "Arguments, venting and frustration — little or no laughter." },
  { id: "deep", label: "Deepest", blurb: "Unusually long and serious for this group." },
  { id: "late_night", label: "Late night", blurb: "Conversations after midnight." },
  { id: "busiest", label: "Busiest", blurb: "The longest single bursts of talking." },
];

const AWARDS: { key: string; title: string; unit: (row: any) => string }[] = [
  { key: "night_owl", title: "Night owl", unit: (r) => `${r.messages_after_midnight} after midnight` },
  { key: "early_bird", title: "Early bird", unit: (r) => `${r.messages_before_8am} before 8am` },
  { key: "fastest_replier", title: "Fastest replier", unit: (r) => `${r.median_minutes} min median` },
  { key: "slowest_replier", title: "Takes their time", unit: (r) => `${r.median_minutes} min median` },
  { key: "biggest_texter", title: "Writes the most", unit: (r) => `${r.avg_words} words avg` },
  { key: "emoji_lover", title: "Emoji lover", unit: (r) => `${r.per_message} per message` },
  { key: "media_sharer", title: "Media machine", unit: (r) => `${r.attachments} attachments` },
  { key: "question_asker", title: "Asks the most", unit: (r) => `${r.pct_of_their_messages}% are questions` },
  { key: "conversation_starter", title: "Starts conversations", unit: (r) => `${r.conversations_started} times` },
  { key: "longest_monologue", title: "Longest monologue", unit: (r) => `${r.messages_in_a_row} in a row` },
  { key: "link_sharer", title: "Link dropper", unit: (r) => `${r.links} links` },
  { key: "ghosted_most", title: "Kept waiting", unit: (r) => `${r.median_wait_minutes} min for a reply` },
];

function Excerpt({ text }: { text: string }) {
  if (!text) return null;
  return (
    <pre
      style={{
        fontFamily: "var(--mono)",
        fontSize: 12,
        background: "var(--bg)",
        border: "1px solid var(--border)",
        borderRadius: 6,
        padding: "10px 12px",
        margin: "10px 0 0",
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        color: "#b9c6d3",
        maxHeight: 320,
        overflowY: "auto",
      }}
    >
      {text}
    </pre>
  );
}

export default function Insights({ archive }: { archive: string | null }) {
  const [data, setData] = useState<any>(null);
  const [kind, setKind] = useState("funny");
  const [moments, setMoments] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .insights(archive)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [archive]);

  useEffect(() => {
    setMoments(null);
    api
      .moments(archive, kind, 5)
      .then(setMoments)
      .catch(() => undefined);
  }, [archive, kind]);

  if (error) return <div className="error-box">{error}</div>;
  if (!data) return <div className="empty">Reading the whole archive...</div>;

  const active = MOMENT_TABS.find((t) => t.id === kind)!;
  const sup = data.superlatives ?? {};
  const rhythms = data.rhythms ?? {};

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="card">
        <h3>Awards</h3>
        <div className="sub">
          Measured across every message, not a sample. Ties go to whoever did it
          more.
        </div>
        <div
          className="grid"
          style={{ gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))" }}
        >
          {AWARDS.map(({ key, title, unit }) => {
            const winner = (sup[key] ?? [])[0];
            if (!winner) return null;
            return (
              <div key={key} className="stat">
                <div className="label">{title}</div>
                <div style={{ fontSize: 16, fontWeight: 600, marginTop: 2 }}>
                  {winner.sender}
                </div>
                <div className="muted" style={{ fontSize: 12.5 }}>
                  {unit(winner)}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="card">
        <h3>Notable conversations</h3>
        <div className="sub">{active.blurb}</div>

        <div className="row" style={{ marginBottom: 4 }}>
          {MOMENT_TABS.map((t) => (
            <button
              key={t.id}
              className={t.id === kind ? "primary" : "ghost"}
              onClick={() => setKind(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {!moments && <div className="empty">Scoring conversations...</div>}

        {moments?.count === 0 && (
          <div className="muted" style={{ padding: "16px 0", fontSize: 13.5 }}>
            {moments.empty_reason}
          </div>
        )}

        {(moments?.moments ?? []).map((m: any) => (
          <div
            key={m.session_id}
            style={{
              borderTop: "1px solid var(--border)",
              paddingTop: 14,
              marginTop: 14,
            }}
          >
            <div className="row" style={{ gap: 14 }}>
              <strong style={{ fontSize: 13.5 }}>
                {String(m.started).replace("T", " ").slice(0, 16)}
              </strong>
              <span className="muted" style={{ fontSize: 12.5 }}>
                {m.messages} messages · {m.people} people
                {m.minutes ? ` · over ${m.minutes} min` : ""}
              </span>
              {m.laughs > 0 && kind === "funny" && (
                <span className="chip good">{m.laughs} laughs</span>
              )}
              {kind === "argument" && (m.heat > 0 || m.apologies > 0) && (
                <span className="chip warn">
                  {m.heat > 0 ? `${m.heat} heated` : ""}
                  {m.heat > 0 && m.apologies > 0 ? " · " : ""}
                  {m.apologies > 0 ? `${m.apologies} apologies` : ""}
                </span>
              )}
              {kind === "deep" && (
                <span className="chip">
                  {m.wordiness_vs_normal}× normal length
                </span>
              )}
              {kind === "late_night" && (
                <span className="chip">{m.night_messages} after midnight</span>
              )}
            </div>
            <Excerpt text={m.excerpt} />
          </div>
        ))}

        {moments?.note && (
          <div className="muted" style={{ fontSize: 12, marginTop: 14 }}>
            {moments.note}
          </div>
        )}
      </div>

      <div className="grid two">
        <div className="card">
          <h3>Longest silences</h3>
          <div className="sub">And who eventually broke them.</div>
          <table>
            <thead>
              <tr>
                <th>Broken by</th>
                <th>On</th>
                <th className="num">Days quiet</th>
              </tr>
            </thead>
            <tbody>
              {(rhythms.longest_silences ?? []).map((r: any, i: number) => (
                <tr key={i}>
                  <td>{r.sender}</td>
                  <td>{String(r.broken_at).slice(0, 10)}</td>
                  <td className="num">{r.days_of_silence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Busiest months</h3>
          <div className="sub">When this group was most alive.</div>
          <table>
            <thead>
              <tr>
                <th>Month</th>
                <th className="num">Messages</th>
                <th className="num">People</th>
              </tr>
            </thead>
            <tbody>
              {(rhythms.busiest_months ?? []).map((r: any) => (
                <tr key={r.year_month}>
                  <td>{r.year_month}</td>
                  <td className="num">{r.messages}</td>
                  <td className="num">{r.active_people}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
