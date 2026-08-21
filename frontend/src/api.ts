export interface ArchiveSummary {
  archive_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  sources: IngestRecord[];
  stats: {
    messages?: number;
    participants?: number;
    media?: number;
    first_message?: string | null;
    last_message?: string | null;
  };
  has_data: boolean;
  size_bytes: number;
}

export interface IngestRecord {
  filename: string;
  ingested_at: string;
  mode: string;
  messages_added: number;
  messages_skipped: number;
  media_added: number;
}

export interface ArchiveDetail extends ArchiveSummary {
  ingested: boolean;
  messages: number;
  participants: number;
  media: number;
  chunks: number;
  embeddings: number;
  semantic_search_ready: boolean;
  api_key_configured: boolean;
  overview: Overview | null;
}

export interface Overview {
  total_messages: number;
  first_message: string | null;
  last_message: string | null;
  participant_count: number;
  active_days: number;
  session_count: number;
  session_gap_hours: number;
  participants: PersonSummary[];
  media: { kind: string; count: number; described: number }[];
}

export interface PersonSummary {
  name: string;
  messages: number;
  initiations: number;
  media_sent: number;
  avg_words: number;
}

/** One kind of attachment still waiting to be described. */
export interface PendingKind {
  kind: string;
  files: number;
  rows: number;
  mb: number;
  requests: number;
  provider: "gemini" | "local-or-groq";
}

/**
 * A job the archive still needs, priced in API requests.
 *
 * `requests` is the honest number: distinct files, not rows, because a photo
 * forwarded four times costs one call.
 */
export interface MaintenanceTask {
  task: "embed" | "describe_media";
  title: string;
  pending: number;
  total: number;
  unit: string;
  requests: number;
  detail: PendingKind[];
  why: string;
  cost_note: string;
  runnable: boolean;
  blocked_reason: string | null;
  warnings: string[];
  /** The model this task actually spends, and what is left of its own budget. */
  model: string | null;
  remaining_today: number | null;
}

export interface MaintenanceSurvey {
  archive_id: string;
  archive_name: string;
  api_key_configured: boolean;
  usage: {
    requests_today: number;
    daily_cap: number | null;
    remaining_today: number | null;
  };
  tasks: MaintenanceTask[];
  complete: string[];
  pending_total: number;
}

export interface LeaderboardRow {
  sender: string;
  messages: number;
  pct: number;
  words: number;
  avg_words: number;
  initiations: number;
  media_sent: number;
  questions: number;
  emojis: number;
  links: number;
  avg_reply_min: number | null;
  median_reply_min: number | null;
  first_seen: string;
  last_seen: string;
}

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  result: unknown;
  error: string | null;
  duration_ms: number;
}

export interface MediaItem {
  media_id: number;
  msg_id: number;
  filename: string;
  kind: string;
  ts: string;
  sender: string | null;
  caption: string | null;
  description: string | null;
  transcript: string | null;
  ocr_text: string | null;
  status: string;
}

export interface MessageRow {
  msg_id: number;
  ts: string;
  sender: string;
  type: string;
  text: string;
}

export interface ProviderInfo {
  id: string;
  label: string;
  base_url: string;
  keys_url: string;
  note: string;
  default_model: string;
}

export interface ChatProvider {
  provider: string;
  label: string;
  model: string;
  base_url: string;
  keys_url: string;
  note: string;
  default_model: string;
  key_set: boolean;
  needs_key: boolean;
}

export interface SpeechConfig {
  provider: string;
  label: string;
  model: string;
  base_url: string;
  key_set: boolean;
  language: string;
  enabled: boolean;
}

export interface AppSettings {
  chat: ChatProvider;
  speech: SpeechConfig;
  speech_providers: ProviderInfo[];
  speech_language: string;
  chat_provider: string;
  chat_base_url: string;
  chat_api_key_set: boolean;
  chat_api_key_hint: string;
  providers: ProviderInfo[];
  api_key_set: boolean;
  api_key_hint: string;
  api_key_source: string;
  chat_model: string;
  vision_model: string;
  embed_model: string;
  embed_dims: number;
  session_gap_hours: number;
  describe_media: boolean;
  transcribe_audio: boolean;
  media_concurrency: number;
  max_requests_per_minute: number;
  max_requests_per_day: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/** Append the archive scope to a query string. */
function scoped(path: string, archive: string | null, extra: Record<string, unknown> = {}) {
  const params = new URLSearchParams();
  if (archive) params.set("archive", archive);
  for (const [k, v] of Object.entries(extra)) {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  }
  const qs = params.toString();
  return qs ? `${path}?${qs}` : path;
}

export const api = {
  // --- archives ---
  listArchives: () =>
    request<{ archives: ArchiveSummary[] }>("/api/archives"),
  createArchive: (name: string) =>
    request<ArchiveSummary>("/api/archives", json({ name })),
  getArchive: (id: string) => request<ArchiveDetail>(`/api/archives/${id}`),
  renameArchive: (id: string, name: string) =>
    request<ArchiveSummary>(`/api/archives/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
  deleteArchive: (id: string) =>
    request<{ deleted: string }>(`/api/archives/${id}`, { method: "DELETE" }),

  // --- settings ---
  getSettings: () => request<AppSettings>("/api/settings"),
  saveSettings: (updates: Partial<Record<string, unknown>>) =>
    request<AppSettings>("/api/settings", json(updates)),
  configCheck: () => request<any>("/api/config/check"),
  usage: () => request<{
    requests_today: number;
    daily_cap: number | null;
    rpm_cap: number | null;
    remaining_today: number | null;
    throttle_events: number;
  }>("/api/usage"),

  // --- data, scoped to one archive ---
  dashboard: (archive: string | null) =>
    request<Record<string, any>>(scoped("/api/stats/dashboard", archive)),
  leaderboard: (archive: string | null) =>
    request<LeaderboardRow[]>(scoped("/api/stats/leaderboard", archive)),
  messages: (archive: string | null, params: Record<string, unknown>) =>
    request<{ total: number; messages: MessageRow[] }>(
      scoped("/api/messages", archive, params),
    ),
  media: (archive: string | null, params: Record<string, unknown>) =>
    request<{ results: MediaItem[]; result_count: number }>(
      scoped("/api/media", archive, params),
    ),
  insights: (archive: string | null) =>
    request<Record<string, any>>(scoped("/api/insights", archive)),
  moments: (archive: string | null, kind: string, limit = 5) =>
    request<Record<string, any>>(
      scoped("/api/insights/moments", archive, { kind, limit }),
    ),
  participants: (archive: string | null) =>
    request<any[]>(scoped("/api/participants", archive)),
  mediaFileUrl: (archive: string | null, mediaId: number) =>
    scoped(`/api/media/${mediaId}/file`, archive),

  // --- maintenance ---
  maintenance: (archive: string | null) =>
    request<MaintenanceSurvey>(scoped("/api/maintenance", archive)),
  runMaintenance: (
    archive: string | null,
    task: MaintenanceTask["task"],
    kinds?: string[],
  ) =>
    request<{ started: boolean; task: string; archive_id: string }>(
      scoped("/api/maintenance/run", archive, {
        task,
        kinds: kinds?.length ? kinds.join(",") : undefined,
      }),
      { method: "POST" },
    ),

  // --- ingest ---
  ingestSample: (name: string) =>
    request<any>(
      `/api/ingest/sample?archive_name=${encodeURIComponent(name)}`,
      { method: "POST" },
    ),
  ingestStatus: (archive: string | null) =>
    request<any>(scoped("/api/ingest/status", archive)),

  uploadExport(
    file: File,
    opts: {
      archive?: string | null;
      archiveName?: string;
      mode: "replace" | "merge";
      describeMedia: boolean;
      embed: boolean;
    },
  ) {
    const form = new FormData();
    form.append("file", file);
    const params = new URLSearchParams({
      mode: opts.mode,
      describe_media: String(opts.describeMedia),
      embed: String(opts.embed),
    });
    if (opts.archive) params.set("archive", opts.archive);
    else if (opts.archiveName) params.set("archive_name", opts.archiveName);

    return fetch(`/api/ingest/upload?${params}`, {
      method: "POST",
      body: form,
    }).then(async (res) => {
      if (!res.ok) {
        let detail = res.statusText;
        try {
          detail = (await res.json()).detail ?? detail;
        } catch {
          /* non-JSON */
        }
        throw new Error(detail);
      }
      return res.json();
    });
  },
};

export type ChatEvent =
  | { type: "tool_call"; name: string; arguments: Record<string, unknown> }
  | ({ type: "tool_result" } & ToolCall)
  | { type: "answer"; text: string; tool_calls: ToolCall[]; steps: number }
  | { type: "error"; message: string };

/**
 * Stream an answer, surfacing each tool call as it happens.
 *
 * The agent can spend several seconds running SQL and searches before it says
 * anything; showing the steps live is the difference between "thinking..." and
 * watching it actually query the database.
 */
export async function streamChat(
  question: string,
  history: { role: string; text: string }[],
  archive: string | null,
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history, archive }),
  });
  if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; a frame may arrive split
    // across chunks, so keep the tail until we see the delimiter.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") return;
      try {
        onEvent(JSON.parse(payload) as ChatEvent);
      } catch {
        // A partial frame is not fatal; the next chunk completes it.
      }
    }
  }
}
