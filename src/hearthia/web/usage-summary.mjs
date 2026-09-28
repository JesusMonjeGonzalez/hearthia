/* Per-conversation usage summary (pure, unit-tested). */

const fmt = (n) => (n == null ? "?" : Number(n).toLocaleString());

export function usageSummary(usage, projection, lastStats) {
  if (!usage || !usage.turns) return null;
  const lines = [
    `Turns: ${fmt(usage.turns)}`,
    `Last input: ${fmt(usage.last_input_tokens)} of ${fmt(usage.allowance_tokens)} tok`,
    `Peak input: ${fmt(usage.peak_input_tokens)} tok`,
    `Totals: ${fmt(usage.total_prompt_tokens)} prompt · ${fmt(usage.total_output_tokens)} output`,
  ];
  if (projection) {
    lines.push(
      `Free before trimming: ${fmt(projection.remaining_tokens)} tok` +
        (projection.growth_tokens_per_turn
          ? ` · growing ~${fmt(Math.round(projection.growth_tokens_per_turn))} tok/turn`
          : ""),
    );
    if (projection.turns_until_trimming != null) {
      lines.push(`Projected turns until the oldest turns are omitted: ${projection.turns_until_trimming}`);
    }
  }
  if (lastStats && lastStats.prompt_n != null) {
    const cached = lastStats.cache_n || 0;
    const total = lastStats.prompt_n + cached;
    const pct = total ? Math.round((100 * cached) / total) : 0;
    lines.push(
      `Last round: prefill ${fmt(lastStats.prompt_n)} tok (${pct}% cached)` +
        (lastStats.predicted_per_second
          ? ` · ${lastStats.predicted_per_second.toFixed(1)} tok/s`
          : ""),
    );
  }
  return lines;
}
