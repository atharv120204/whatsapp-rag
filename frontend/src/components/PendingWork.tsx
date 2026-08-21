import { useCallback, useEffect, useRef, useState } from "react";
import { api, type MaintenanceSurvey, type MaintenanceTask } from "../api";

/**
 * Offers the archive's unfinished work where the user actually is.
 *
 * Embedding and describing attachments were previously CLI-only, which in
 * practice meant they did not get done. The hard part is not the button, it is
 * quoting the cost honestly enough that "yes" is an informed answer: the number
 * shown is distinct files rather than rows, it moves as kinds are ticked off,
 * and it is stated against how much of today's budget is actually left.
 */

interface Props {
  archive: string | null;
  /** Bump the app's data after a job finishes, so counts refresh. */
  onCompleted: () => void;
}

const GEMINI_VISION_PER_DAY = 20;

const KIND_LABEL: Record<string, string> = {
  image: "Photos",
  sticker: "Stickers",
  video: "Video",
  document: "Documents",
  voice: "Voice notes",
  audio: "Audio",
  contact: "Contact cards",
};

function requestsFor(task: MaintenanceTask, selected: Set<string>): number {
  if (task.task !== "describe_media") return task.requests;
  return task.detail
    .filter((d) => selected.has(d.kind))
    .reduce((total, d) => total + d.requests, 0);
}

function filesFor(task: MaintenanceTask, selected: Set<string>): number {
  if (task.task !== "describe_media") return task.pending;
  return task.detail
    .filter((d) => selected.has(d.kind))
    .reduce((total, d) => total + d.files, 0);
}

/**
 * Warnings for the *current* selection.
 *
 * Recomputed here rather than taken from the survey because deselecting
 * stickers can turn "about three weeks of runs" into "this afternoon", and a
 * warning that does not move with the checkboxes teaches people to ignore it.
 */
function warningsFor(
  task: MaintenanceTask,
  requests: number,
  remaining: number | null,
): string[] {
  const out: string[] = [];

  // A budget of zero is stated as a blocked reason instead, which is both
  // clearer and stops the button pretending there is something to do.
  if (remaining !== null && remaining > 0 && requests > remaining) {
    out.push(
      `Today's budget has ${remaining} request${remaining === 1 ? "" : "s"} left, ` +
        `so this gets through roughly ${remaining} of ${requests} and then stops ` +
        `cleanly. Finished work is cached — running it again tomorrow costs nothing extra.`,
    );
  }

  if (task.task === "describe_media" && requests > GEMINI_VISION_PER_DAY) {
    const days = Math.max(1, Math.round(requests / GEMINI_VISION_PER_DAY));
    out.push(
      `Gemini's free tier allows roughly ${GEMINI_VISION_PER_DAY} vision requests a day, ` +
        `so ${requests.toLocaleString()} files is on the order of ${days} day${days === 1 ? "" : "s"} ` +
        `of runs. Untick what you do not need.`,
    );
  }

  return out;
}

export default function PendingWork({ archive, onCompleted }: Props) {
  const [survey, setSurvey] = useState<MaintenanceSurvey | null>(null);
  const [selection, setSelection] = useState<Record<string, Set<string>>>({});
  const [confirming, setConfirming] = useState<string | null>(null);
  const [running, setRunning] = useState<any>(null);
  const [finished, setFinished] = useState<string | null>(null);
  /** Ran out of budget: the designed ending, not a failure. */
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const pollRef = useRef<number | null>(null);

  const load = useCallback(() => {
    if (!archive) return;
    api
      .maintenance(archive)
      .then((s) => {
        setSurvey(s);
        // Default to everything pending, but leave the choice visible: on a
        // group chat stickers can be half the files and rarely worth describing.
        setSelection((current) => {
          const next = { ...current };
          for (const task of s.tasks) {
            if (task.task === "describe_media" && !next[task.task]) {
              next[task.task] = new Set(task.detail.map((d) => d.kind));
            }
          }
          return next;
        });
      })
      .catch(() => setSurvey(null));
  }, [archive]);

  useEffect(() => {
    setDismissed(false);
    setFinished(null);
    setConfirming(null);
    load();
  }, [archive, load]);

  // Poll while a job runs. Shared with ingest on purpose: one writer per
  // archive, so this also notices an upload started on the Add chat tab.
  useEffect(() => {
    if (!running || !archive) return;
    function poll() {
      api
        .ingestStatus(archive)
        .then((state) => {
          setRunning(state.running ? state : null);
          if (!state.running) {
            if (pollRef.current) window.clearInterval(pollRef.current);
            pollRef.current = null;
            setFinished(state.result?.summary?.message ?? "Finished.");
            setNotice(state.notice ?? null);
            setError(state.error ?? null);
            load();
            onCompleted();
          }
        })
        .catch(() => undefined);
    }
    pollRef.current = window.setInterval(poll, 900);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [running, archive, load, onCompleted]);

  async function start(task: MaintenanceTask) {
    setError(null);
    setNotice(null);
    setFinished(null);
    setConfirming(null);
    const kinds =
      task.task === "describe_media"
        ? [...(selection[task.task] ?? new Set<string>())]
        : undefined;
    try {
      await api.runMaintenance(archive, task.task, kinds);
      setRunning({ stage: "queued", message: "Starting", detail: {} });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  function toggleKind(task: string, kind: string) {
    setSelection((current) => {
      const next = new Set(current[task] ?? []);
      if (next.has(kind)) next.delete(kind);
      else next.add(kind);
      return { ...current, [task]: next };
    });
  }

  if (!survey || dismissed) return null;
  if (!survey.tasks.length && !finished) return null;

  if (running) {
    const pct = running.detail?.pct ?? 0;
    return (
      <div className="card" style={{ marginBottom: 16 }}>
        <h3>Working</h3>
        <div className="sub">{running.message ?? "Starting..."}</div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>
          You can leave this tab — it keeps going. Stopping the server stops the
          job, and everything finished by then is kept.
        </div>
      </div>
    );
  }

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="row">
        <h3 style={{ margin: 0 }}>
          {survey.tasks.length
            ? "This archive has unfinished work"
            : "Up to date"}
        </h3>
        <div className="spacer" />
        <button className="ghost" onClick={() => setDismissed(true)}>
          Not now
        </button>
      </div>

      {finished && (
        <div
          className="muted"
          style={{ fontSize: 13, marginTop: 10, color: "var(--accent)" }}
        >
          {finished}
        </div>
      )}
      {notice && (
        <ul className="warn-list">
          <li>{notice}</li>
        </ul>
      )}
      {error && (
        <div className="error-box" style={{ marginTop: 12 }}>
          {error}
        </div>
      )}

      {survey.tasks.map((task) => {
        const selected = selection[task.task] ?? new Set<string>();
        // Per task, not per app: embedding and vision have separate budgets
        // whose real limits differ by two orders of magnitude, so one figure
        // for both is how a spent vision quota advertises embedding capacity.
        const remaining = task.remaining_today ?? null;
        const requests = requestsFor(task, selected);
        const files = filesFor(task, selected);
        const warnings = warningsFor(task, requests, remaining);
        const isConfirming = confirming === task.task;
        const nothingPicked = task.task === "describe_media" && files === 0;

        return (
          <div
            key={task.task}
            style={{
              marginTop: 16,
              paddingTop: 16,
              borderTop: "1px solid var(--border)",
            }}
          >
            <div style={{ fontSize: 14.5, marginBottom: 4 }}>
              {task.title}
              <span className="muted" style={{ fontSize: 12.5, marginLeft: 8 }}>
                {task.pending.toLocaleString()} {task.unit} pending
              </span>
            </div>
            <div className="muted" style={{ fontSize: 13, lineHeight: 1.5 }}>
              {task.why}
            </div>

            {task.detail.length > 0 && (
              <div className="row" style={{ gap: 16, marginTop: 12 }}>
                {task.detail.map((d) => (
                  <label key={d.kind} className="row" style={{ gap: 7 }}>
                    <input
                      type="checkbox"
                      checked={selected.has(d.kind)}
                      onChange={() => toggleKind(task.task, d.kind)}
                    />
                    <span style={{ fontSize: 13 }}>
                      {KIND_LABEL[d.kind] ?? d.kind}{" "}
                      <span className="muted">
                        ({d.files.toLocaleString()}
                        {d.provider === "gemini"
                          ? ""
                          : " — free, not Gemini"}
                        )
                      </span>
                    </span>
                  </label>
                ))}
              </div>
            )}

            <div
              className="muted"
              style={{ fontSize: 12.5, marginTop: 12, fontFamily: "var(--mono)" }}
            >
              {requests.toLocaleString()} API request
              {requests === 1 ? "" : "s"}
              {remaining !== null && (
                <>
                  {" · "}
                  {remaining.toLocaleString()} left in today's budget
                </>
              )}
            </div>

            {warnings.length > 0 && (
              <ul className="warn-list">
                {warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            )}

            {task.blocked_reason ? (
              <div className="error-box" style={{ marginTop: 12 }}>
                {task.blocked_reason}
              </div>
            ) : isConfirming ? (
              <div style={{ marginTop: 14 }}>
                <div style={{ fontSize: 13.5, marginBottom: 10 }}>
                  Run this now? It will make about{" "}
                  <strong>{requests.toLocaleString()}</strong> Gemini request
                  {requests === 1 ? "" : "s"}
                  {task.task === "describe_media" && (
                    <>
                      {" "}
                      across {files.toLocaleString()} file
                      {files === 1 ? "" : "s"}
                    </>
                  )}
                  . It stops on its own when the budget runs out, and resumes
                  where it left off next time.
                </div>
                <div className="row">
                  <button onClick={() => start(task)}>Yes, run it</button>
                  <button className="ghost" onClick={() => setConfirming(null)}>
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="row" style={{ marginTop: 14 }}>
                <button
                  onClick={() => setConfirming(task.task)}
                  disabled={!task.runnable || nothingPicked}
                >
                  {nothingPicked ? "Pick something first" : task.title}
                </button>
              </div>
            )}
          </div>
        );
      })}

      {survey.complete.length > 0 && (
        <div className="muted" style={{ fontSize: 12.5, marginTop: 16 }}>
          {survey.complete.join(" · ")}
        </div>
      )}
    </div>
  );
}
