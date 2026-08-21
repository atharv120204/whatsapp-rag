import { useEffect, useState } from "react";
import { api, type MediaItem } from "../api";

const KIND_ICON: Record<string, string> = {
  image: "🖼", video: "🎬", voice: "🎙", audio: "🎵",
  sticker: "🏷", document: "📄", contact: "👤", other: "📎",
};

const KINDS = ["", "image", "video", "voice", "audio", "sticker", "document"];

export default function Media({ archive }: { archive: string | null }) {
  const [items, setItems] = useState<MediaItem[]>([]);
  const [kind, setKind] = useState("");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    api
      .media(archive, { kind, q: query, limit: 120 })
      .then((r) => setItems(r.results))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
      .finally(() => setLoading(false));
  }, [archive, kind, query]);

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="card">
        <h3>Shared media</h3>
        <div className="sub">
          Searches what is <em>inside</em> each file — what a photo shows, what a
          voice note says, text read out of an image.
        </div>
        <div className="row">
          <input
            type="search"
            placeholder="Search descriptions, transcripts, text in images..."
            value={query}
            style={{ flex: 1, minWidth: 260 }}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            {KINDS.map((k) => (
              <option key={k} value={k}>
                {k === "" ? "All types" : k}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}
      {loading && <div className="empty">Loading...</div>}

      {!loading && items.length === 0 && (
        <div className="empty">
          No media found. If your export was made without "Attach media", the
          files are not in it — re-export with media to search them.
        </div>
      )}

      <div className="media-grid">
        {items.map((item) => (
          <div className="media-card" key={item.media_id}>
            <div className="media-thumb">
              {item.kind === "image" || item.kind === "sticker" ? (
                <img
                  src={api.mediaFileUrl(archive, item.media_id)}
                  alt={item.description ?? item.filename}
                  loading="lazy"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = "none";
                  }}
                />
              ) : (
                <span>{KIND_ICON[item.kind] ?? KIND_ICON.other}</span>
              )}
            </div>
            <div className="media-meta">
              <div className="who">
                {item.sender ?? "unknown"} · {String(item.ts).slice(0, 10)} ·{" "}
                {item.kind}
              </div>
              {item.caption && (
                <div style={{ marginBottom: 6 }}>"{item.caption}"</div>
              )}
              {item.description ? (
                <div className="desc">{item.description}</div>
              ) : (
                <div className="muted">
                  {item.status === "skipped"
                    ? "Not describable"
                    : item.status === "error"
                      ? "Description failed"
                      : "Not described yet"}
                </div>
              )}
              {item.transcript && (
                <div className="transcript">"{item.transcript}"</div>
              )}
              {item.ocr_text && (
                <div className="muted" style={{ marginTop: 6, fontSize: 11.5 }}>
                  Text in image: {item.ocr_text.slice(0, 160)}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
