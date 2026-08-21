import { useState } from "react";
import { api, type ArchiveSummary } from "../api";

interface Props {
  archives: ArchiveSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onChanged: () => void;
  onLoadChat: () => void;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

export default function Archives({
  archives,
  activeId,
  onSelect,
  onChanged,
  onLoadChat,
}: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [confirmText, setConfirmText] = useState("");

  async function run(id: string, fn: () => Promise<unknown>) {
    setBusy(id);
    setError(null);
    try {
      await fn();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="grid" style={{ gap: 16, maxWidth: 900, margin: "0 auto" }}>
      <div className="card">
        <h3>Archives on this device</h3>
        <div className="sub">
          Each chat is a separate database with its own media. Nothing is shared
          between them.
        </div>
        <button className="primary" onClick={onLoadChat}>
          Add a chat export
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {archives.length === 0 && (
        <div className="empty">
          No archives yet. Add a chat export to get started.
        </div>
      )}

      {archives.map((archive) => {
        const isActive = archive.archive_id === activeId;
        const deleting = confirmDelete === archive.archive_id;

        return (
          <div
            className="card"
            key={archive.archive_id}
            style={
              isActive
                ? { borderColor: "var(--accent-dim)" }
                : undefined
            }
          >
            <div className="row" style={{ alignItems: "flex-start" }}>
              <div style={{ flex: 1, minWidth: 220 }}>
                {renaming === archive.archive_id ? (
                  <div className="row">
                    <input
                      type="text"
                      value={draftName}
                      autoFocus
                      onChange={(e) => setDraftName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter")
                          run(archive.archive_id, async () => {
                            await api.renameArchive(archive.archive_id, draftName);
                            setRenaming(null);
                          });
                        if (e.key === "Escape") setRenaming(null);
                      }}
                    />
                    <button
                      className="ghost"
                      onClick={() =>
                        run(archive.archive_id, async () => {
                          await api.renameArchive(archive.archive_id, draftName);
                          setRenaming(null);
                        })
                      }
                    >
                      Save
                    </button>
                    <button className="ghost" onClick={() => setRenaming(null)}>
                      Cancel
                    </button>
                  </div>
                ) : (
                  <h3 style={{ marginBottom: 6 }}>
                    {archive.name}
                    {isActive && (
                      <span className="chip good" style={{ marginLeft: 10 }}>
                        active
                      </span>
                    )}
                  </h3>
                )}

                <div className="muted" style={{ fontSize: 12.5 }}>
                  {archive.stats.messages
                    ? `${archive.stats.messages.toLocaleString()} messages · ${
                        archive.stats.participants ?? 0
                      } people · ${archive.stats.media ?? 0} attachments`
                    : "Empty — no export loaded yet"}
                </div>
                <div className="muted" style={{ fontSize: 12, marginTop: 3 }}>
                  {archive.stats.first_message
                    ? `${String(archive.stats.first_message).slice(0, 10)} to ${String(
                        archive.stats.last_message,
                      ).slice(0, 10)} · `
                    : ""}
                  {formatBytes(archive.size_bytes)} on disk
                </div>
              </div>

              <div className="row">
                {!isActive && (
                  <button
                    className="ghost"
                    onClick={() => onSelect(archive.archive_id)}
                  >
                    Open
                  </button>
                )}
                <button
                  className="ghost"
                  onClick={() => {
                    setRenaming(archive.archive_id);
                    setDraftName(archive.name);
                  }}
                >
                  Rename
                </button>
                <button
                  className="ghost"
                  disabled={busy === archive.archive_id}
                  onClick={() => {
                    setConfirmDelete(deleting ? null : archive.archive_id);
                    setConfirmText("");
                  }}
                >
                  Delete
                </button>
              </div>
            </div>

            {archive.sources.length > 0 && (
              <details className="trace" style={{ marginTop: 12 }}>
                <summary>
                  {archive.sources.length} export
                  {archive.sources.length === 1 ? "" : "s"} loaded
                </summary>
                <div className="trace-body">
                  <table>
                    <thead>
                      <tr>
                        <th>File</th>
                        <th>Mode</th>
                        <th className="num">Added</th>
                        <th className="num">Skipped</th>
                        <th>When</th>
                      </tr>
                    </thead>
                    <tbody>
                      {archive.sources.map((s, i) => (
                        <tr key={i}>
                          <td>{s.filename}</td>
                          <td>{s.mode}</td>
                          <td className="num">{s.messages_added}</td>
                          <td className="num">{s.messages_skipped}</td>
                          <td className="muted" style={{ fontSize: 12 }}>
                            {s.ingested_at.replace("T", " ").slice(0, 16)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}

            {deleting && (
              <div
                className="error-box"
                style={{ marginTop: 12, display: "grid", gap: 10 }}
              >
                <div>
                  This permanently deletes <strong>{archive.name}</strong>, its
                  database and all {archive.stats.media ?? 0} media files. It
                  cannot be undone.
                </div>
                <div className="row">
                  <input
                    type="text"
                    placeholder={`Type "${archive.name}" to confirm`}
                    value={confirmText}
                    style={{ flex: 1, minWidth: 200 }}
                    onChange={(e) => setConfirmText(e.target.value)}
                  />
                  <button
                    className="primary"
                    disabled={
                      confirmText !== archive.name || busy === archive.archive_id
                    }
                    onClick={() =>
                      run(archive.archive_id, async () => {
                        await api.deleteArchive(archive.archive_id);
                        setConfirmDelete(null);
                      })
                    }
                  >
                    Delete permanently
                  </button>
                  <button className="ghost" onClick={() => setConfirmDelete(null)}>
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
