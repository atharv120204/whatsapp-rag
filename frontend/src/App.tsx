import { useCallback, useEffect, useState } from "react";
import { api, type ArchiveDetail, type ArchiveSummary } from "./api";
import Archives from "./components/Archives";
import Browse from "./components/Browse";
import Chat from "./components/Chat";
import Dashboard from "./components/Dashboard";
import Ingest from "./components/Ingest";
import Insights from "./components/Insights";
import Media from "./components/Media";
import PendingWork from "./components/PendingWork";
import Settings from "./components/Settings";

type Tab = "chat" | "dashboard" | "insights" | "media" | "browse" | "archives" | "load" | "settings";

const TABS: { id: Tab; label: string; needsData?: boolean }[] = [
  { id: "chat", label: "Ask", needsData: true },
  { id: "dashboard", label: "Dashboard", needsData: true },
  { id: "insights", label: "Insights", needsData: true },
  { id: "media", label: "Media", needsData: true },
  { id: "browse", label: "Browse", needsData: true },
  { id: "archives", label: "Archives" },
  { id: "load", label: "Add chat" },
  { id: "settings", label: "Settings" },
];

const ACTIVE_KEY = "chatarchive.active";

export default function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [archives, setArchives] = useState<ArchiveSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(
    () => localStorage.getItem(ACTIVE_KEY),
  );
  const [detail, setDetail] = useState<ArchiveDetail | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [loaded, setLoaded] = useState(false);

  const loadArchives = useCallback(async () => {
    const { archives: list } = await api.listArchives();
    setArchives(list);
    setLoaded(true);

    // Keep the selection valid: a remembered archive may have been deleted.
    setActiveId((current) => {
      if (current && list.some((a) => a.archive_id === current)) return current;
      return list[0]?.archive_id ?? null;
    });
    return list;
  }, []);

  useEffect(() => {
    loadArchives().catch(() => setLoaded(true));
  }, [loadArchives]);

  useEffect(() => {
    if (!activeId) {
      setDetail(null);
      return;
    }
    localStorage.setItem(ACTIVE_KEY, activeId);
    api
      .getArchive(activeId)
      .then(setDetail)
      .catch(() => setDetail(null));
  }, [activeId, reloadKey]);

  const refresh = useCallback(
    (archiveId?: string) => {
      loadArchives().then(() => {
        if (archiveId) setActiveId(archiveId);
        setReloadKey((k) => k + 1);
      });
    },
    [loadArchives],
  );

  // Land somewhere useful rather than on an empty chat screen.
  useEffect(() => {
    if (!loaded) return;
    if (!archives.length) setTab("load");
    else if (detail && !detail.ingested) setTab("load");
  }, [loaded, archives.length, detail?.ingested]);

  const hasData = Boolean(detail?.ingested);
  const ready = Boolean(detail?.ingested && detail?.api_key_configured);

  const blockedReason = !detail
    ? "No archive selected. Add a chat export to get started."
    : !detail.ingested
      ? `"${detail.name}" is empty. Load an export into it from the Add chat tab.`
      : !detail.api_key_configured
        ? "No Gemini API key configured, so the chat agent cannot run. Add one on the Settings tab. The Dashboard, Media and Browse tabs work without it."
        : undefined;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          chat<span>.</span>archive
        </div>

        {archives.length > 0 && (
          <select
            className="archive-picker"
            value={activeId ?? ""}
            onChange={(e) => setActiveId(e.target.value)}
            title="Which chat you are looking at"
          >
            {archives.map((a) => (
              <option key={a.archive_id} value={a.archive_id}>
                {a.name}
              </option>
            ))}
          </select>
        )}

        <nav className="tabs">
          {TABS.map((t) => (
            <button
              key={t.id}
              className={`tab${tab === t.id ? " active" : ""}`}
              disabled={t.needsData && !hasData}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>

        <div className="status-chips">
          {detail?.ingested ? (
            <span className="chip good">
              {detail.messages.toLocaleString()} messages
            </span>
          ) : (
            <span className="chip warn">no chat loaded</span>
          )}
          {detail?.media ? (
            <span className="chip">{detail.media} attachments</span>
          ) : null}
          {detail && !detail.api_key_configured && (
            <span className="chip bad">no API key</span>
          )}
          {detail?.ingested && !detail.semantic_search_ready && (
            <span className="chip warn">keyword search only</span>
          )}
        </div>
      </header>

      <main className={`content${tab === "chat" ? " chat-mode" : ""}`}>
        {/*
          Offered on the tabs where the gap is visible -- a photo with no
          description on Media, a paraphrase that finds nothing on Dashboard --
          and kept off Chat, where a banner above the conversation would be in
          the way of the thing the user came to do.
        */}
        {hasData && (tab === "dashboard" || tab === "media" || tab === "insights") && (
          <PendingWork archive={activeId} onCompleted={() => refresh()} />
        )}
        {tab === "chat" && (
          <Chat
            key={activeId ?? "none"}
            archive={activeId}
            ready={ready}
            blockedReason={blockedReason}
          />
        )}
        {tab === "dashboard" && (
          <Dashboard key={`${activeId}-${reloadKey}`} archive={activeId} />
        )}
        {tab === "insights" && (
          <Insights key={`${activeId}-${reloadKey}`} archive={activeId} />
        )}
        {tab === "media" && (
          <Media key={`${activeId}-${reloadKey}`} archive={activeId} />
        )}
        {tab === "browse" && (
          <Browse key={`${activeId}-${reloadKey}`} archive={activeId} />
        )}
        {tab === "archives" && (
          <Archives
            archives={archives}
            activeId={activeId}
            onSelect={(id) => {
              setActiveId(id);
              setTab("dashboard");
            }}
            onChanged={() => refresh()}
            onLoadChat={() => setTab("load")}
          />
        )}
        {tab === "load" && (
          <Ingest
            archives={archives}
            onDone={(id) => {
              refresh(id);
              setTab("dashboard");
            }}
          />
        )}
        {tab === "settings" && <Settings onSaved={() => refresh()} />}
      </main>
    </div>
  );
}
