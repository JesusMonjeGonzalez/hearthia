/* Turn an API error body into the sentence a human should read. */

export function errorMessage(text, fallback = "request failed") {
  const raw = String(text || "").trim();
  if (!raw) return fallback;
  try {
    const data = JSON.parse(raw);
    const candidate = data?.detail ?? data?.error ?? data?.message;
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
    if (candidate != null) return JSON.stringify(candidate);
  } catch {
    /* not JSON: it is already a message */
  }
  return raw;
}
