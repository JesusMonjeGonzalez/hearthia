/* api.js — shared utilities and API client for Hearthia dashboard. */

export const $ = (s) => document.querySelector(s);
export const GB = 1024 ** 3;

export const fmtGB = (b) => (b / GB).toFixed(1) + " GB";

export const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export const fmtClock = (s) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
};

export const STATE_WORDS = {
  ready: "warm",
  starting: "kindling",
  stopping: "cooling",
  stopped: "cold",
};

export const stateLabel = (state) => STATE_WORDS[state] || state || "cold";

marked.setOptions({ gfm: true, breaks: false });
export const renderMD = (text) => DOMPurify.sanitize(marked.parse(text || ""));

export function highlightIn(el) {
  el.querySelectorAll("pre").forEach((pre) => {
    if (pre.parentElement?.classList.contains("codewrap")) return;
    const wrap = document.createElement("div");
    wrap.className = "codewrap";
    pre.replaceWith(wrap);
    wrap.appendChild(pre);
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "copy";
    btn.addEventListener("click", () => {
      navigator.clipboard.writeText(pre.textContent);
      btn.textContent = "copied";
      setTimeout(() => (btn.textContent = "copy"), 1200);
    });
    wrap.appendChild(btn);
    pre.querySelectorAll("code").forEach((c) => hljs.highlightElement(c));
  });
}
