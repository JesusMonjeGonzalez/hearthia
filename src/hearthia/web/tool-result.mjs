/* Decode structured tool results without ever treating output as HTML. */
export function describeTool(message) {
  let result;
  try { result = JSON.parse(message.content); } catch { /* ordinary text tool */ }
  if (result && typeof result === "object" && result.kind === "edit" && typeof result.diff === "string") {
    const broken = typeof result.syntax_error === "string";
    return { kind: "edit", failed: result.ok === false || broken,
      title:
        `Edit · ${result.path} · ${result.changed ? "changed" : "unchanged"}` +
        (result.diff_truncated ? " · diff truncated" : "") +
        (broken ? ` · SYNTAX ERROR (${result.syntax_error})` : ""),
      body: result.diff || "No content change.",
      detail: broken
        ? `Syntax check failed: ${result.syntax_error}. The file was written; fix it before running anything.`
        : `SHA-256: ${result.after_sha256 || "unknown"}` };
  }
  if (result && typeof result === "object" && result.kind === "command" && typeof result.output === "string") {
    const command = Array.isArray(result.argv) ? result.argv.join(" ").slice(0, 300) : "command";
    return { kind: "command", failed: result.ok !== true,
      title: `${result.ok ? "Passed" : "Failed"} · ${command} · exit ${result.exit_code} · ${result.duration_seconds}s`,
      body: result.output || "(no output)",
      detail: `${result.cwd || ""}${result.limit ? ` · stopped: ${result.limit}` : ""}${result.output_truncated ? " · output truncated" : ""}` };
  }
  if (result && typeof result === "object" && result.kind === "plan" &&
      (Array.isArray(result.steps) || Array.isArray(result.done))) {
    // A status-only update carries no step list (compact ack): show progress
    // from done/pending alone instead of falling back to a generic card.
    const steps = Array.isArray(result.steps) ? result.steps : null;
    const done = new Set(Array.isArray(result.done) ? result.done : []);
    const total = steps ? steps.length : done.size + (result.pending || 0);
    const lines = steps
      ? steps.map((step, index) => `${done.has(index + 1) ? "✔" : "○"} ${step}`)
      : ["(steps unchanged)"];
    return { kind: "plan", failed: result.ok === false,
      title: `Plan · ${done.size}/${total} done` +
        (result.pending ? ` · ${result.pending} pending` : "") +
        (result.warning ? " · unverified edits" : ""),
      body: lines.join("\n"), detail: result.warning || "" };
  }
  if (result && typeof result === "object" && result.kind === "job") {
    if (!result.ok) return { kind: "job", failed: true, title: `Job · refused`,
      body: String(result.error || "unknown"), detail: "" };
    if (Array.isArray(result.jobs)) {
      const lines = result.jobs.map((job) =>
        `${job.state.padEnd(7)} ${job.id}  ${job.duration_seconds}s  ${job.argv}`);
      return { kind: "job", failed: false, title: `Jobs · ${result.running} running`,
        body: lines.join("\n") || "(none yet)", detail: "" };
    }
    const failed = result.state === "failed" || result.state === "timeout" || result.state === "killed";
    return { kind: "job", failed,
      title: `Job ${result.id} · ${result.state}` +
        (result.exit_code != null ? ` · exit ${result.exit_code}` : "") +
        ` · ${result.duration_seconds}s`,
      body: [result.argv, result.tail || ""].filter(Boolean).join("\n\n"),
      detail: result.log_truncated ? "log truncated (size cap)" : "" };
  }
  if (result && typeof result === "object" && result.kind === "mcp") {
    return { kind: "mcp", failed: result.ok === false,
      title: `MCP ${result.server}/${result.tool} · ${result.ok ? "ok" : "error"}`,
      body: String(result.output || ""), detail: "" };
  }
  const text = String(message.content || "");
  const subagent = /^Subagent report \(([^)]*)\):\n?([\s\S]*)$/.exec(text);
  if (subagent) {
    return { kind: "task", failed: false, title: `Subagent · ${subagent[1]}`, body: subagent[2], detail: "" };
  }
  return { kind: "text", failed: text.startsWith("Error:"),
    title: `Tool result · ${message.tool_name || message.tool_call_id || ""}`, body: text, detail: "" };
}
