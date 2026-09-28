/* Incremental SSE decoding shared by chat and its Node regression tests. */
export class ChatStream {
  constructor() {
    this.decoder = new TextDecoder();
    this.buffer = "";
  }
  feed(bytes) {
    this.buffer += this.decoder.decode(bytes, { stream: true });
    if (this.buffer.length > 1_048_576) throw new Error("Chat event exceeded the stream limit");
    const lines = this.buffer.split("\n");
    this.buffer = lines.pop();
    const events = [];
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") { events.push({ done: true }); continue; }
      try { events.push(JSON.parse(payload)); } catch { /* ignore malformed events */ }
    }
    return events;
  }
}
