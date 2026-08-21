import { useEffect, useState } from "react";
import { api, type AppSettings } from "../api";

export default function Settings({ onSaved }: { onSaved: () => void }) {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [key, setKey] = useState("");
  const [chatKey, setChatKey] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [check, setCheck] = useState<any>(null);
  const [usage, setUsage] = useState<any>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    api.getSettings().then(setSettings).catch(() => undefined);
    api.usage().then(setUsage).catch(() => undefined);
  }, []);

  async function save(updates: Record<string, unknown>) {
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      setSettings(await api.saveSettings(updates));
      setNotice("Saved.");
      onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  async function verify() {
    setChecking(true);
    setCheck(null);
    try {
      setCheck(await api.configCheck());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setChecking(false);
    }
  }

  if (!settings) return <div className="empty">Loading...</div>;

  return (
    <div className="grid" style={{ gap: 16, maxWidth: 760, margin: "0 auto" }}>
      <div className="card">
        <h3>Chat provider</h3>
        <div className="sub">
          Which model answers your questions and writes the SQL. Free-tier
          limits differ enormously between providers, so this is worth choosing
          deliberately. Embeddings and media understanding always use Gemini —
          most other hosts serve no embeddings endpoint at all.
        </div>

        <div className="row" style={{ marginBottom: 12 }}>
          <label style={{ width: 120, fontSize: 13.5 }}>Provider</label>
          <select
            value={settings.chat_provider}
            style={{ minWidth: 200 }}
            onChange={(e) => save({ chat_provider: e.target.value })}
          >
            {settings.providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
          {settings.chat.keys_url && (
            <a
              href={settings.chat.keys_url}
              target="_blank"
              rel="noreferrer"
              style={{ color: "var(--blue)", fontSize: 13 }}
            >
              get a key
            </a>
          )}
        </div>

        <div className="muted" style={{ fontSize: 12.5, marginBottom: 12 }}>
          {settings.chat.note}
        </div>

        {settings.chat.needs_key && settings.chat_provider !== "gemini" && (
          <div className="row" style={{ marginBottom: 12 }}>
            <label style={{ width: 120, fontSize: 13.5 }}>API key</label>
            <input
              type="password"
              placeholder={
                settings.chat_api_key_set
                  ? `Key set (${settings.chat_api_key_hint}) — type a new one to replace`
                  : `Paste your ${settings.chat.label} key`
              }
              value={chatKey}
              style={{ flex: 1, minWidth: 240 }}
              onChange={(e) => setChatKey(e.target.value)}
            />
            <button
              className="primary"
              disabled={saving || !chatKey.trim()}
              onClick={() => {
                save({ chat_api_key: chatKey.trim() });
                setChatKey("");
              }}
            >
              Save
            </button>
          </div>
        )}

        <div className="row" style={{ marginBottom: 12 }}>
          <label style={{ width: 120, fontSize: 13.5 }}>Model</label>
          <input
            type="text"
            key={settings.chat.model}
            defaultValue={settings.chat.model}
            style={{ flex: 1, minWidth: 220 }}
            onBlur={(e) => {
              if (e.target.value.trim() !== settings.chat.model)
                save({ chat_model: e.target.value.trim() });
            }}
          />
          <button className="ghost" onClick={verify} disabled={checking}>
            {checking ? "Checking..." : "Test"}
          </button>
        </div>

        {settings.chat_provider === "custom" && (
          <div className="row">
            <label style={{ width: 120, fontSize: 13.5 }}>Base URL</label>
            <input
              type="text"
              defaultValue={settings.chat_base_url}
              placeholder="https://host/v1"
              style={{ flex: 1, minWidth: 240 }}
              onBlur={(e) => save({ chat_base_url: e.target.value.trim() })}
            />
          </div>
        )}

        {check?.chat && (
          <div style={{ marginTop: 12 }}>
            {check.chat.ok ? (
              <>
                <span className="chip good">{check.chat.provider} reachable</span>
                {check.chat.models?.length > 0 && (
                  <>
                    <div
                      className="muted"
                      style={{ fontSize: 12.5, margin: "10px 0 6px" }}
                    >
                      {check.chat.model_available === false
                        ? `"${settings.chat.model}" is not in this key's model list. Pick one below.`
                        : `${check.chat.models.length} models available to this key:`}
                    </div>
                    <select
                      value={settings.chat.model}
                      style={{ minWidth: 280 }}
                      onChange={(e) => save({ chat_model: e.target.value })}
                    >
                      {!check.chat.models.includes(settings.chat.model) && (
                        <option value={settings.chat.model}>
                          {settings.chat.model} (not listed)
                        </option>
                      )}
                      {check.chat.models.map((m: string) => (
                        <option key={m} value={m}>
                          {m}
                        </option>
                      ))}
                    </select>
                  </>
                )}
                {check.chat.models_error && (
                  <div className="muted" style={{ fontSize: 12.5, marginTop: 8 }}>
                    Could not list models: {check.chat.models_error}
                  </div>
                )}
              </>
            ) : (
              <div className="error-box">{check.chat.error}</div>
            )}
          </div>
        )}
      </div>

      <div className="card">
        <h3>Gemini API key</h3>
        <div className="sub">
          Powers semantic search and media understanding, and the chat
          agent too if Gemini is selected above. Ingest, the dashboard,
          keyword search and browsing work without it. Get a key at{" "}
          <a
            href="https://aistudio.google.com/apikey"
            target="_blank"
            rel="noreferrer"
            style={{ color: "var(--blue)" }}
          >
            aistudio.google.com/apikey
          </a>
          .
        </div>

        <div className="row">
          <input
            type="password"
            placeholder={
              settings.api_key_set
                ? `Key set (${settings.api_key_hint}) — type a new one to replace`
                : "Paste your API key"
            }
            value={key}
            style={{ flex: 1, minWidth: 260 }}
            onChange={(e) => setKey(e.target.value)}
          />
          <button
            className="primary"
            disabled={saving || !key.trim()}
            onClick={() => {
              save({ gemini_api_key: key.trim() });
              setKey("");
            }}
          >
            Save key
          </button>
          <button className="ghost" onClick={verify} disabled={checking}>
            {checking ? "Checking..." : "Test"}
          </button>
        </div>

        <div className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>
          Stored in plain text in <code>data/config.json</code> on this machine,
          the same exposure as a .env file. It is never sent to the browser.
          {settings.api_key_source === "env" &&
            " Currently coming from the environment; saving here overrides it."}
        </div>

        {check?.gemini && (
          <div style={{ marginTop: 12 }}>
            {check.gemini.error ? (
              <div className="error-box">{check.gemini.error}</div>
            ) : (
              <>
                <div className="muted" style={{ fontSize: 12.5 }}>
                  {check.gemini.model_count} models visible to this key.
                </div>
                <table style={{ marginTop: 8 }}>
                  <tbody>
                    {[...(check.gemini.available ?? []), ...(check.gemini.missing ?? [])].map(
                      (entry: any) => {
                        const ok = (check.gemini.available ?? []).includes(entry);
                        return (
                          <tr key={entry.role}>
                            <td style={{ width: 120 }}>{entry.role}</td>
                            <td>
                              <code>{entry.model}</code>
                            </td>
                            <td style={{ width: 90 }}>
                              <span className={`chip ${ok ? "good" : "bad"}`}>
                                {ok ? "available" : "missing"}
                              </span>
                            </td>
                          </tr>
                        );
                      },
                    )}
                  </tbody>
                </table>
              </>
            )}
          </div>
        )}
      </div>

      <div className="card">
        <h3>Gemini models</h3>
        <div className="sub">
          Used by Gemini only, for reading attachments and building search
          vectors. The chat model is set in the Chat provider card above.
        </div>
        {(
          [
            ["vision_model", "Images, audio, video"],
            ["embed_model", "Embeddings"],
          ] as const
        ).map(([field, label]) => (
          <div className="row" key={field} style={{ marginBottom: 10 }}>
            <label style={{ width: 180, fontSize: 13.5 }}>{label}</label>
            <input
              type="text"
              defaultValue={settings[field]}
              style={{ flex: 1, minWidth: 220 }}
              onBlur={(e) => {
                if (e.target.value.trim() !== settings[field])
                  save({ [field]: e.target.value.trim() });
              }}
            />
          </div>
        ))}
        <div className="muted" style={{ fontSize: 12.5 }}>
          Embedding width is {settings.embed_dims}. Changing it requires
          re-ingesting, because stored vectors are at the old width.
        </div>
      </div>

      <div className="card">
        <h3>Conversations</h3>
        <div className="sub">
          Silence longer than this starts a new conversation, which is what
          defines "who initiates". One to two hours measures conversational
          bursts; six to eight measures who texts first each day. The chatbot
          can override it per question.
        </div>
        <div className="row">
          <input
            type="number"
            step="0.5"
            min="0.25"
            defaultValue={settings.session_gap_hours}
            style={{ width: 110 }}
            onBlur={(e) => {
              const value = parseFloat(e.target.value);
              if (!Number.isNaN(value) && value !== settings.session_gap_hours)
                save({ session_gap_hours: value });
            }}
          />
          <span className="muted" style={{ fontSize: 13.5 }}>
            hours of silence ends a conversation
          </span>
        </div>
        <div className="muted" style={{ fontSize: 12.5, marginTop: 10 }}>
          Changing this affects new ingests. Existing archives keep the value
          they were built with until re-ingested.
        </div>
      </div>


      <div className="card">
        <h3>Staying inside the free tier</h3>
        <div className="sub">
          Google sets free-tier limits per project and shows them in{" "}
          <a
            href="https://aistudio.google.com/rate-limit"
            target="_blank"
            rel="noreferrer"
            style={{ color: "var(--blue)" }}
          >
            AI Studio
          </a>
          , not in its docs — so set these to match what your account actually
          allows. Requests are spaced to fit, and back off further if Google
          pushes back.
        </div>

        <div className="row" style={{ marginBottom: 10 }}>
          <label style={{ width: 180, fontSize: 13.5 }}>Requests / minute</label>
          <input
            type="number"
            min="0"
            defaultValue={settings.max_requests_per_minute}
            style={{ width: 100 }}
            onBlur={(e) => {
              const v = parseInt(e.target.value, 10);
              if (!Number.isNaN(v) && v !== settings.max_requests_per_minute)
                save({ max_requests_per_minute: v });
            }}
          />
          <span className="muted" style={{ fontSize: 12.5 }}>0 = no limit</span>
        </div>

        <div className="row" style={{ marginBottom: 10 }}>
          <label style={{ width: 180, fontSize: 13.5 }}>Requests / day</label>
          <input
            type="number"
            min="0"
            defaultValue={settings.max_requests_per_day}
            style={{ width: 100 }}
            onBlur={(e) => {
              const v = parseInt(e.target.value, 10);
              if (!Number.isNaN(v) && v !== settings.max_requests_per_day)
                save({ max_requests_per_day: v });
            }}
          />
          {usage && (
            <span className="muted" style={{ fontSize: 12.5 }}>
              {usage.requests_today} used today
              {usage.remaining_today !== null
                ? ` · ${usage.remaining_today} left`
                : ""}
            </span>
          )}
        </div>

        <div className="muted" style={{ fontSize: 12.5 }}>
          When the daily budget runs out, the ingest stops cleanly rather than
          failing. Everything processed is cached by file content, so running it
          again tomorrow continues from exactly where it stopped and re-pays for
          nothing.
        </div>
      </div>

      <div className="card">
        <h3>Voice notes</h3>
        <div className="sub">
          Which service transcribes audio. Whisper is better at speech than a
          general multimodal model, and on Groq it is free — which matters,
          because voice notes are usually the largest group of attachments and
          Gemini's daily allowance is small.
        </div>

        <div className="row" style={{ marginBottom: 12 }}>
          <label style={{ width: 130, fontSize: 13.5 }}>Transcribed by</label>
          <select
            value={settings.speech.provider}
            style={{ minWidth: 200 }}
            onChange={(e) => save({ speech_provider: e.target.value })}
          >
            <option value="gemini">Gemini (default)</option>
            {settings.speech_providers.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
          {settings.speech.enabled && settings.speech.key_set && (
            <span className="chip good">key found</span>
          )}
        </div>

        {settings.speech.enabled && (
          <>
            <div className="row" style={{ marginBottom: 12 }}>
              <label style={{ width: 130, fontSize: 13.5 }}>Model</label>
              <input
                type="text"
                key={settings.speech.model}
                defaultValue={settings.speech.model}
                style={{ flex: 1, minWidth: 200 }}
                onBlur={(e) => {
                  if (e.target.value.trim() !== settings.speech.model)
                    save({ speech_model: e.target.value.trim() });
                }}
              />
            </div>

            <div className="row" style={{ marginBottom: 12 }}>
              <label style={{ width: 130, fontSize: 13.5 }}>Language</label>
              <input
                type="text"
                placeholder="auto-detect"
                key={settings.speech.language}
                defaultValue={settings.speech.language}
                style={{ width: 120 }}
                onBlur={(e) => save({ speech_language: e.target.value.trim() })}
              />
              <span className="muted" style={{ fontSize: 12.5 }}>
                two-letter code, e.g. <code>hi</code>, <code>en</code>
              </span>
            </div>

            <div className="muted" style={{ fontSize: 12.5 }}>
              Worth setting. Auto-detection is unreliable on short, code-mixed
              clips — a few seconds of Hindi was read as Tagalog until the
              language was named.
            </div>
          </>
        )}

        {check?.speech && (
          <div style={{ marginTop: 12 }}>
            {check.speech.ok ? (
              <span className="chip good">
                {check.speech.model} available
              </span>
            ) : (
              <div className="muted" style={{ fontSize: 12.5 }}>
                {check.speech.error ??
                  `${check.speech.model} not found. Available: ${
                    (check.speech.speech_models ?? []).join(", ") || "none"
                  }`}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="card">
        <h3>Media processing</h3>
        <div className="sub">
          The only part that costs money on a large archive. Results are cached
          by file content, so the same photo is never described twice.
        </div>
        <label className="row" style={{ gap: 8, marginBottom: 10 }}>
          <input
            type="checkbox"
            checked={settings.describe_media}
            onChange={(e) => save({ describe_media: e.target.checked })}
          />
          <span style={{ fontSize: 13.5 }}>
            Describe photos, videos and documents
          </span>
        </label>
        <label className="row" style={{ gap: 8, marginBottom: 10 }}>
          <input
            type="checkbox"
            checked={settings.transcribe_audio}
            onChange={(e) => save({ transcribe_audio: e.target.checked })}
          />
          <span style={{ fontSize: 13.5 }}>Transcribe voice notes and audio</span>
        </label>
        <div className="row">
          <label style={{ width: 180, fontSize: 13.5 }}>Parallel requests</label>
          <input
            type="number"
            min="1"
            max="16"
            defaultValue={settings.media_concurrency}
            style={{ width: 90 }}
            onBlur={(e) => {
              const value = parseInt(e.target.value, 10);
              if (!Number.isNaN(value) && value !== settings.media_concurrency)
                save({ media_concurrency: value });
            }}
          />
          <span className="muted" style={{ fontSize: 12.5 }}>
            lower this if you hit rate limits
          </span>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}
      {notice && (
        <div className="muted" style={{ fontSize: 13 }}>
          {notice}
        </div>
      )}
    </div>
  );
}
