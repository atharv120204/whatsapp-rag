import { useEffect, useRef, useState } from "react";
import { api, type ArchiveSummary } from "../api";

interface Props {
  archives: ArchiveSummary[];
  onDone: (archiveId: string) => void;
}

const STAGE_ORDER = [
  "unpack", "parse", "merge", "participants", "enrich", "load",
  "media", "chunk", "embed", "done",
];

type Target = { kind: "new"; name: string } | { kind: "existing"; id: string };

export default function Ingest({ archives, onDone }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [describeMedia, setDescribeMedia] = useState(true);
  const [embed, setEmbed] = useState(true);
  const [mode, setMode] = useState<"replace" | "merge">("merge");
  // Defaults to a new archive on purpose. Pre-selecting the open archive
  // would mean a stray drop merges someone else's chat into it.
  const [targetId, setTargetId] = useState<string>("__new__");
  const [newName, setNewName] = useState("");
  const [state, setState] = useState<any>(null);
  const [watching, setWatching] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<number | null>(null);

  const target: Target =
    targetId === "__new__"
      ? { kind: "new", name: newName }
      : { kind: "existing", id: targetId };

  const existing = archives.find((a) => a.archive_id === targetId);
  const existingCount = existing?.stats.messages ?? 0;
  const running = Boolean(state?.running) || uploading;

  useEffect(() => {
    if (!watching) return;
    function poll() {
      api
        .ingestStatus(watching)
        .then((s) => {
          setState(s);
          if (!s.running) {
            if (pollRef.current) window.clearInterval(pollRef.current);
            pollRef.current = null;
            if (s.result?.ok && watching) onDone(watching);
          }
        })
        .catch(() => undefined);
    }
    poll();
    pollRef.current = window.setInterval(poll, 900);
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [watching, onDone]);

  async function upload(file: File) {
    if (target.kind === "new" && !target.name.trim()) {
      setError("Give the new archive a name first.");
      return;
    }
    setError(null);
    setUploading(true);
    setState(null);
    try {
      const res = await api.uploadExport(file, {
        archive: target.kind === "existing" ? target.id : null,
        archiveName: target.kind === "new" ? target.name.trim() : undefined,
        mode: target.kind === "new" ? "replace" : mode,
        describeMedia,
        embed,
      });
      setWatching(res.archive_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  }

  async function runSample() {
    setError(null);
    setState(null);
    try {
      const res = await api.ingestSample(newName.trim() || "Sample group chat");
      setWatching(res.archive_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const stageIndex = STAGE_ORDER.indexOf(state?.stage ?? "");
  const pct =
    state?.detail?.pct ??
    (stageIndex >= 0 ? ((stageIndex + 1) / STAGE_ORDER.length) * 100 : 0);

  const result = state?.result;
  const stages = result?.stages ?? {};
  const merge = stages.merge;

  return (
    <div className="grid" style={{ gap: 16, maxWidth: 860, margin: "0 auto" }}>
      <div className="card">
        <h3>Add a chat export</h3>
        <div className="sub">
          In WhatsApp: open the chat, tap the menu, <em>More</em> →{" "}
          <em>Export chat</em> → <strong>Attach media</strong>.
        </div>

        <div className="row" style={{ marginBottom: 14 }}>
          <label style={{ width: 90, fontSize: 13.5 }}>Load into</label>
          <select
            value={targetId}
            onChange={(e) => setTargetId(e.target.value)}
            style={{ minWidth: 220 }}
          >
            <option value="__new__">＋ A new archive</option>
            {archives.map((a) => (
              <option key={a.archive_id} value={a.archive_id}>
                {a.name} ({(a.stats.messages ?? 0).toLocaleString()} messages)
              </option>
            ))}
          </select>
          {target.kind === "new" && (
            <input
              type="text"
              placeholder="Name it, e.g. Family Group"
              value={newName}
              style={{ flex: 1, minWidth: 200 }}
              onChange={(e) => setNewName(e.target.value)}
            />
          )}
        </div>

        {target.kind === "existing" && existingCount > 0 && (
          <div style={{ marginBottom: 14 }}>
            <div className="row" style={{ gap: 18 }}>
              <label className="row" style={{ gap: 7 }}>
                <input
                  type="radio"
                  checked={mode === "merge"}
                  onChange={() => setMode("merge")}
                />
                <span style={{ fontSize: 13.5 }}>Merge into it</span>
              </label>
              <label className="row" style={{ gap: 7 }}>
                <input
                  type="radio"
                  checked={mode === "replace"}
                  onChange={() => setMode("replace")}
                />
                <span style={{ fontSize: 13.5 }}>Replace it</span>
              </label>
            </div>
            <div
              className={mode === "replace" ? "error-box" : "muted"}
              style={{ fontSize: 12.5, marginTop: 10 }}
            >
              {mode === "merge" ? (
                <>
                  Messages already in <strong>{existing?.name}</strong> are
                  skipped, so re-uploading the same export changes nothing. Use
                  this to combine a with-media export (WhatsApp caps those at
                  roughly the last 10,000 messages) with a full-history
                  text-only one — photos are matched across both, not
                  duplicated.
                </>
              ) : (
                <>
                  This deletes all {existingCount.toLocaleString()} messages
                  currently in <strong>{existing?.name}</strong> and its media
                  before loading. Cannot be undone.
                </>
              )}
            </div>
          </div>
        )}

        <div
          className={`dropzone${dragOver ? " over" : ""}`}
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            const file = e.dataTransfer.files?.[0];
            if (file) upload(file);
          }}
        >
          <div style={{ fontSize: 15, marginBottom: 6 }}>
            Drop your export here, or click to browse
          </div>
          <div className="muted" style={{ fontSize: 13 }}>
            .zip with media, or a plain .txt transcript
          </div>
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".zip,.txt"
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) upload(file);
            e.target.value = "";
          }}
        />

        <div className="row" style={{ marginTop: 16 }}>
          <label className="row" style={{ gap: 7 }}>
            <input
              type="checkbox"
              checked={describeMedia}
              onChange={(e) => setDescribeMedia(e.target.checked)}
            />
            <span style={{ fontSize: 13.5 }}>
              Describe photos, voice notes and video
            </span>
          </label>
          <label className="row" style={{ gap: 7 }}>
            <input
              type="checkbox"
              checked={embed}
              onChange={(e) => setEmbed(e.target.checked)}
            />
            <span style={{ fontSize: 13.5 }}>Build semantic search index</span>
          </label>
          <div className="spacer" />
          <button className="ghost" onClick={runSample} disabled={running}>
            Try a sample chat
          </button>
        </div>

        <div className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>
          Both options call the Gemini API. Turn them off for a fast, free
          ingest — every statistic still works, only semantic search and media
          understanding are skipped.
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}

      {running && (
        <div className="card">
          <h3>
            {uploading ? "Uploading" : `Working: ${state?.stage ?? "starting"}`}
          </h3>
          <div className="sub">{state?.message ?? "Preparing..."}</div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${Math.min(pct, 100)}%` }}
            />
          </div>
          {state?.detail?.total ? (
            <div className="muted" style={{ fontSize: 12.5, marginTop: 8 }}>
              {state.detail.done} of {state.detail.total}
              {state.detail.cached ? ` · ${state.detail.cached} from cache` : ""}
              {state.detail.errors ? ` · ${state.detail.errors} failed` : ""}
            </div>
          ) : null}
        </div>
      )}

      {state?.error && !running && <div className="error-box">{state.error}</div>}

      {result && !running && (
        <div className="card">
          <h3>{result.ok ? "Ingest complete" : "Ingest failed"}</h3>
          <div className="sub">
            {result.archive_name} · {result.mode} · {result.elapsed_seconds}s
          </div>

          {result.mode === "merge" && merge && (
            <div className="grid stats" style={{ marginBottom: 14 }}>
              <div className="stat">
                <div className="value">{merge.added}</div>
                <div className="label">new messages</div>
              </div>
              <div className="stat">
                <div className="value">{merge.skipped}</div>
                <div className="label">already present</div>
              </div>
              <div className="stat">
                <div className="value">{merge.upgraded}</div>
                <div className="label">media filled in</div>
              </div>
              <div className="stat">
                <div className="value">{merge.total}</div>
                <div className="label">total now</div>
              </div>
            </div>
          )}

          <div className="grid stats" style={{ marginBottom: 14 }}>
            <div className="stat">
              <div className="value">{stages.parse?.parsed_messages ?? 0}</div>
              <div className="label">in this export</div>
            </div>
            <div className="stat">
              <div className="value">{stages.participants?.count ?? 0}</div>
              <div className="label">people</div>
            </div>
            <div className="stat">
              <div className="value">{stages.media?.media_rows ?? 0}</div>
              <div className="label">attachments</div>
            </div>
            <div className="stat">
              <div className="value">
                {stages.media?.understanding?.done ?? 0}
              </div>
              <div className="label">media described</div>
            </div>
            <div className="stat">
              <div className="value">
                {(stages.embeddings?.embedded ?? 0) +
                  (stages.embeddings?.cached ?? 0)}
              </div>
              <div className="label">vectors</div>
            </div>
          </div>

          {stages.parse && (
            <div className="muted" style={{ fontSize: 12.5 }}>
              Date format detected: {stages.parse.date_order} (
              {stages.parse.date_order_confidence}) ·{" "}
              {stages.parse.system_messages} system notices ·{" "}
              {stages.parse.media_messages} media ·{" "}
              {stages.parse.deleted_messages} deleted
              {stages.embeddings?.cached
                ? ` · ${stages.embeddings.cached} embeddings reused from cache`
                : ""}
            </div>
          )}

          {result.warnings?.length > 0 && (
            <ul className="warn-list">
              {result.warnings.map((w: string, i: number) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          )}
          {result.errors?.length > 0 && (
            <div className="error-box" style={{ marginTop: 12 }}>
              {result.errors.join(" ")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
