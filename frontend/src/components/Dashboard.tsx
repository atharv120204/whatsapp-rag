import { Fragment, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type LeaderboardRow } from "../api";

const PALETTE = [
  "#25d366", "#58a6ff", "#f0b429", "#c678dd", "#e06c75",
  "#56b6c2", "#98c379", "#d19a66",
];

const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

const axis = { stroke: "#8b9aab", fontSize: 11 };
const tooltipStyle = {
  contentStyle: {
    background: "#182029",
    border: "1px solid #232e3a",
    borderRadius: 8,
    fontSize: 12,
  },
};

function Stat({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="card stat">
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  );
}

/** Hour-by-weekday grid. A heatmap reads faster here than 168 bars. */
function Heatmap({ data }: { data: { weekday: number; hour: number; messages: number }[] }) {
  const max = Math.max(1, ...data.map((d) => d.messages));
  const lookup = new Map(data.map((d) => [`${d.weekday}-${d.hour}`, d.messages]));

  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ display: "grid", gridTemplateColumns: "34px repeat(24, 1fr)", gap: 2, minWidth: 560 }}>
        <div />
        {Array.from({ length: 24 }, (_, h) => (
          <div key={h} style={{ fontSize: 9, color: "var(--muted)", textAlign: "center" }}>
            {h % 3 === 0 ? h : ""}
          </div>
        ))}
        {WEEKDAYS.map((day, d) => (
          <Fragment key={day}>
            <div style={{ fontSize: 10.5, color: "var(--muted)", lineHeight: "20px" }}>
              {day}
            </div>
            {Array.from({ length: 24 }, (_, h) => {
              const n = lookup.get(`${d}-${h}`) ?? 0;
              return (
                <div
                  key={`${d}-${h}`}
                  title={`${day} ${h}:00 — ${n} messages`}
                  style={{
                    height: 20,
                    borderRadius: 3,
                    background: n
                      ? `rgba(37, 211, 102, ${0.12 + 0.88 * (n / max)})`
                      : "#151c24",
                  }}
                />
              );
            })}
          </Fragment>
        ))}
      </div>
    </div>
  );
}

export default function Dashboard({ archive }: { archive: string | null }) {
  const [data, setData] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .dashboard(archive)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [archive]);

  if (error) return <div className="error-box">{error}</div>;
  if (!data) return <div className="empty">Loading...</div>;

  const leaderboard: LeaderboardRow[] = data.leaderboard ?? [];
  if (!leaderboard.length)
    return <div className="empty">No data yet. Ingest a chat export first.</div>;

  const colorFor = (name: string) =>
    PALETTE[leaderboard.findIndex((r) => r.sender === name) % PALETTE.length];

  const totalMessages = leaderboard.reduce((sum, r) => sum + r.messages, 0);
  const totalMedia = leaderboard.reduce((sum, r) => sum + r.media_sent, 0);

  // Pivot the per-person timeline into one row per period for a multi-series line.
  const timelineByPeriod = new Map<string, Record<string, number | string>>();
  for (const row of data.timeline ?? []) {
    const entry: Record<string, number | string> =
      timelineByPeriod.get(row.period) ?? { period: row.period };
    entry[row.sender] = row.messages;
    timelineByPeriod.set(row.period, entry);
  }
  const timeline = [...timelineByPeriod.values()];

  const hourly = Array.from({ length: 24 }, (_, h) => {
    const entry: Record<string, number | string> = { hour: `${h}` };
    for (const row of data.hourly ?? []) {
      if (row.hour === h) entry[row.sender] = row.messages;
    }
    return entry;
  });

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid stats">
        <Stat value={totalMessages.toLocaleString()} label="Messages" />
        <Stat value={leaderboard.length} label="People" />
        <Stat value={(data.initiation?.initiators ?? []).reduce((s: number, r: any) => s + r.initiations, 0)} label="Conversations" />
        <Stat value={totalMedia.toLocaleString()} label="Attachments" />
        <Stat
          value={data.streaks?.longest_streaks?.[0]?.days ?? 0}
          label="Longest daily streak"
        />
      </div>

      <div className="card">
        <h3>Who sent what</h3>
        <div className="sub">
          Exact counts from the database, not a sample.
        </div>
        <table>
          <thead>
            <tr>
              <th>Person</th>
              <th className="num">Messages</th>
              <th className="num">Share</th>
              <th className="num">Started</th>
              <th className="num">Questions</th>
              <th className="num">Media</th>
              <th className="num">Avg words</th>
              <th className="num">Median reply</th>
            </tr>
          </thead>
          <tbody>
            {leaderboard.map((row) => (
              <tr key={row.sender}>
                <td>
                  <span
                    style={{
                      display: "inline-block",
                      width: 8,
                      height: 8,
                      borderRadius: 2,
                      background: colorFor(row.sender),
                      marginRight: 8,
                    }}
                  />
                  {row.sender}
                </td>
                <td className="num">{row.messages.toLocaleString()}</td>
                <td className="num">{row.pct}%</td>
                <td className="num">{row.initiations}</td>
                <td className="num">{row.questions}</td>
                <td className="num">{row.media_sent}</td>
                <td className="num">{row.avg_words}</td>
                <td className="num">
                  {row.median_reply_min != null ? `${row.median_reply_min}m` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid two">
        <div className="card">
          <h3>Who starts conversations</h3>
          <div className="sub">
            First message after {data.initiation?.gap_hours ?? 4}+ hours of silence.
          </div>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={data.initiation?.initiators ?? []} layout="vertical"
                      margin={{ left: 10, right: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#232e3a" horizontal={false} />
              <XAxis type="number" {...axis} />
              <YAxis dataKey="sender" type="category" width={110} {...axis} />
              <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
              <Bar dataKey="initiations" radius={[0, 4, 4, 0]}>
                {(data.initiation?.initiators ?? []).map((row: any) => (
                  <Cell key={row.sender} fill={colorFor(row.sender)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3>Who ends them</h3>
          <div className="sub">Last word before the silence.</div>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={data.initiation?.enders ?? []} layout="vertical"
                      margin={{ left: 10, right: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#232e3a" horizontal={false} />
              <XAxis type="number" {...axis} />
              <YAxis dataKey="sender" type="category" width={110} {...axis} />
              <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
              <Bar dataKey="conversations_ended" radius={[0, 4, 4, 0]}>
                {(data.initiation?.enders ?? []).map((row: any) => (
                  <Cell key={row.sender} fill={colorFor(row.sender)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h3>Activity over time</h3>
        <div className="sub">Messages per month, per person.</div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={timeline}>
            <CartesianGrid strokeDasharray="3 3" stroke="#232e3a" />
            <XAxis dataKey="period" {...axis} />
            <YAxis {...axis} />
            <Tooltip {...tooltipStyle} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {leaderboard.map((row) => (
              <Line
                key={row.sender}
                type="monotone"
                dataKey={row.sender}
                stroke={colorFor(row.sender)}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid two">
        <div className="card">
          <h3>When this group talks</h3>
          <div className="sub">Darker means busier. Hover for counts.</div>
          <Heatmap data={data.heatmap ?? []} />
        </div>

        <div className="card">
          <h3>By hour of day</h3>
          <div className="sub">Stacked per person.</div>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={hourly}>
              <CartesianGrid strokeDasharray="3 3" stroke="#232e3a" />
              <XAxis dataKey="hour" {...axis} />
              <YAxis {...axis} />
              <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
              {leaderboard.map((row) => (
                <Bar
                  key={row.sender}
                  dataKey={row.sender}
                  stackId="a"
                  fill={colorFor(row.sender)}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid two">
        <div className="card">
          <h3>Who replies to whom</h3>
          <div className="sub">
            Direct replies and typical response time. Reveals sub-groups.
          </div>
          <table>
            <thead>
              <tr>
                <th>Replies to</th>
                <th>Responder</th>
                <th className="num">Count</th>
                <th className="num">Median</th>
              </tr>
            </thead>
            <tbody>
              {(data.responses ?? []).slice(0, 12).map((row: any, i: number) => (
                <tr key={i}>
                  <td>{row.responding_to}</td>
                  <td>{row.responder}</td>
                  <td className="num">{row.replies}</td>
                  <td className="num">{row.median_min}m</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Notable days</h3>
          <div className="sub">Busiest days and longest silences.</div>
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th className="num">Messages</th>
              </tr>
            </thead>
            <tbody>
              {(data.streaks?.busiest_days ?? []).slice(0, 6).map((row: any) => (
                <tr key={row.date}>
                  <td>{row.date}</td>
                  <td className="num">{row.messages}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="sub" style={{ marginTop: 16, marginBottom: 6 }}>
            Longest silences
          </div>
          <table>
            <thead>
              <tr>
                <th>Broken by</th>
                <th>On</th>
                <th className="num">Days quiet</th>
              </tr>
            </thead>
            <tbody>
              {(data.streaks?.longest_silences ?? []).slice(0, 5).map((row: any, i: number) => (
                <tr key={i}>
                  <td>{row.sender}</td>
                  <td>{String(row.resumed_at).slice(0, 10)}</td>
                  <td className="num">{row.silent_days}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
