/* Bounded local log history; pausing freezes a snapshot, not the connection. */
export class LogBuffer {
  constructor(limit = 200000) {
    this.limit = limit;
    this.text = "";
    this.snapshot = null;
  }

  append(chunk) {
    this.text = (this.text + chunk).slice(-this.limit);
  }

  setPaused(paused) {
    this.snapshot = paused ? this.text : null;
  }

  clear() {
    this.text = "";
    if (this.snapshot !== null) this.snapshot = "";
  }

  visible(query = "") {
    const text = this.snapshot ?? this.text;
    const needle = query.trim().toLowerCase();
    return needle ? text.split("\n").filter(line => line.toLowerCase().includes(needle)).join("\n") : text;
  }
}
