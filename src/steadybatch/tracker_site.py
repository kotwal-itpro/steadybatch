"""Build the tracker's public page: one static HTML file plus the raw data.

The page is plain HTML, CSS and a little JavaScript with no outside libraries,
so GitHub Pages can serve it as is. The data is embedded in the page and also
written next to it as runs.jsonl and runs.csv for anyone who wants it.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "target", "label", "provider", "model", "state", "submitted_at", "submit_hour_utc", "created_at",
    "started_at", "ended_at", "turnaround_s", "turnaround_source", "queue_s", "records", "ok",
    "missing", "error", "invalid", "truncated", "compared_with_previous", "changed_vs_previous",
    "exact_accuracy", "cost_usd", "cost_per_1000_usd", "model_versions", "batch_id", "error_message",
]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    fields = list(dict.fromkeys(CSV_FIELDS))
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({**r, "model_versions": " ".join(r.get("model_versions") or [])})
    return buf.getvalue()


def build(runs_path: Path, out: Path, *, open_path: Path | None = None,
          now: datetime | None = None) -> Path:
    rows = _rows(runs_path)
    open_batches = json.loads(open_path.read_text()) if open_path and open_path.exists() else []
    now = now or datetime.now(timezone.utc)
    out.mkdir(parents=True, exist_ok=True)
    if runs_path.exists():
        shutil.copyfile(runs_path, out / "runs.jsonl")
    (out / "runs.csv").write_text(_csv(rows))
    slim_open = [{k: b.get(k) for k in ("target", "provider", "model", "label", "submitted_ts")}
                 for b in open_batches]
    data = {"rows": rows, "open": slim_open, "built_at": now.isoformat(timespec="minutes")}
    html = PAGE.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    page = out / "index.html"
    page.write_text(html)
    return page


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Batch API Tracker</title>
<meta name="description" content="Measured turnaround, lost lines and answer drift for the OpenAI, Gemini and Anthropic batch APIs, run twice a day.">
<style>
:root{color-scheme:light;
 --surface:#fcfcfb;--surface-2:#f3f2ef;--grid:#e4e3df;--text:#0b0b0b;--text-2:#52514e;--muted:#7a7975;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--bad:#c4302b;--warn:#a86b00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
 --surface:#1a1a19;--surface-2:#242422;--grid:#33332f;--text:#f0efec;--text-2:#c3c2b7;--muted:#97968e;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--bad:#f07b72;--warn:#e0a63a}}
:root[data-theme="dark"]{color-scheme:dark;
 --surface:#1a1a19;--surface-2:#242422;--grid:#33332f;--text:#f0efec;--text-2:#c3c2b7;--muted:#97968e;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--bad:#f07b72;--warn:#e0a63a}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:18px;margin:40px 0 4px}
p{margin:6px 0;color:var(--text-2)}
.lede{font-size:16px;max-width:760px}
.meta{font-size:13px;color:var(--muted)}
a{color:var(--s1)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:20px}
.tile{background:var(--surface-2);border-radius:10px;padding:14px 16px}
.tile .name{display:flex;align-items:center;gap:8px;font-weight:600}
.key{width:10px;height:10px;border-radius:50%;flex:none}
.tile dl{display:grid;grid-template-columns:1fr auto;gap:2px 12px;margin:10px 0 0;font-size:14px}
.tile dt{color:var(--text-2)}.tile dd{margin:0;font-variant-numeric:tabular-nums;font-weight:600;text-align:right}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--text-2);margin:8px 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.chart{position:relative;margin-top:4px}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{fill:var(--text-2);font-size:12px}
.tip{position:absolute;pointer-events:none;background:var(--text);color:var(--surface);font-size:12.5px;
 padding:6px 9px;border-radius:6px;white-space:nowrap;transform:translate(-50%,-110%);opacity:0;transition:opacity .08s}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid);white-space:nowrap}
th{color:var(--text-2);font-weight:600}
td.num,th.num{text-align:right}
.flag{font-weight:600}.flag.bad{color:var(--bad)}.flag.warn{color:var(--warn)}
.empty{padding:18px;background:var(--surface-2);border-radius:10px;color:var(--text-2)}
details{margin-top:8px}summary{cursor:pointer;color:var(--text-2)}
ul{color:var(--text-2);padding-left:20px}li{margin:4px 0}
</style></head>
<body><main>
<h1>Batch API Tracker</h1>
<p class="lede">Batch APIs promise results "within 24 hours". This page measures what actually happens: the same 200 requests go to each provider's batch API twice a day, and every run records how long it took, whether every line came back, and how many answers changed since the last run.</p>
<p class="meta" id="updated"></p>

<div class="tiles" id="tiles"></div>

<h2>How long a batch takes</h2>
<p>Each dot is one batch, from the provider's own created and finished timestamps. The scale is logarithmic.</p>
<div class="legend" id="legend1"></div>
<div class="chart" id="turn"></div>

<h2>By time of submission</h2>
<p>Median and slowest turnaround over the last 30 days, by the hour (UTC) the batch was sent.</p>
<div class="wrap"><table id="slots"></table></div>

<h2>Answers that changed since the previous run</h2>
<p>The inputs never change and temperature is 0, so any change comes from the provider side.</p>
<div class="legend" id="legend2"></div>
<div class="chart" id="drift"></div>

<h2>Incidents</h2>
<p>Runs where a batch failed or timed out, lines were lost or came back broken, a batch took over 6 hours, or more than 5% of answers changed.</p>
<div class="wrap"><table id="incidents"></table></div>

<h2>All runs</h2>
<p>Download the data: <a href="runs.csv">runs.csv</a> · <a href="runs.jsonl">runs.jsonl</a></p>
<details><summary>Show the table</summary><div class="wrap"><table id="all"></table></div></details>

<h2>How it works</h2>
<ul>
<li>200 synthetic support conversations with known answers, the same every run. Each model fills in four fields under a JSON Schema (<a href="https://github.com/kotwal-itpro/steadybatch/tree/main/tracker">inputs and settings</a>).</li>
<li>Batches go out at 01:00 and 13:00 UTC from GitHub Actions, using <a href="https://github.com/kotwal-itpro/steadybatch">steadybatch</a>. A second job checks every 30 minutes and records each finished batch.</li>
<li>Turnaround is the provider's finished time minus its created time. Lost lines are requests with no result at all. Nothing is resubmitted.</li>
<li>Limits: 200-request batches may queue differently from very large jobs, and one account per provider can't tell you about every region or tier.</li>
<li>Cost is computed from token counts and the published batch prices in <a href="https://github.com/kotwal-itpro/steadybatch/blob/main/tracker/prices.json">prices.json</a>.</li>
</ul>
<p class="meta">Built by <a href="https://github.com/kotwal-itpro">Ankur Kotwal</a>. Code and data are open; corrections welcome as issues.</p>
</main>
<script>
const DATA = /*DATA*/null;
const NAMES = {openai:"OpenAI", gemini:"Gemini", anthropic:"Anthropic"};
const rows = DATA.rows.slice().sort((a,b)=>a.submitted_ts-b.submitted_ts);
const targets = [...new Set(rows.map(r=>r.target).concat(DATA.open.map(o=>o.target)))];
const colorOf = t => `var(--s${(targets.indexOf(t)%3)+1})`;
const label = t => { const r = rows.slice().reverse().find(x=>x.target===t) || DATA.open.find(x=>x.target===t);
  return r.label || `${NAMES[r.provider]||r.provider} · ${r.model}`; };
const DAY = 86400;
const nowTs = Date.parse(DATA.built_at)/1000;
const recent = rows.filter(r => r.submitted_ts > nowTs - 30*DAY);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function dur(s){ if(s==null) return "–"; if(s<90) return Math.round(s)+" s";
  if(s<5400) return Math.round(s/60)+" min"; return (s/3600).toFixed(s<36000?1:0)+" h"; }
function pct(x){ return x==null ? "–" : (x*100).toFixed(x<0.1?1:0)+"%"; }
function median(a){ if(!a.length) return null; const s=a.slice().sort((x,y)=>x-y), m=s.length>>1;
  return s.length%2 ? s[m] : (s[m-1]+s[m])/2; }
function when(ts){ return new Date(ts*1000).toISOString().slice(0,16).replace("T"," ")+" UTC"; }

document.getElementById("updated").textContent =
  `Updated ${DATA.built_at.replace("T"," ").slice(0,16)} UTC · ${rows.length} runs recorded` +
  (rows.length ? ` since ${when(rows[0].submitted_ts).slice(0,10)}` : "");

function legend(id){ document.getElementById(id).innerHTML = targets.map(t =>
  `<span><i class="key" style="background:${colorOf(t)}"></i>${esc(label(t))}</span>`).join(""); }

// Tiles: one per provider, last 30 days.
document.getElementById("tiles").innerHTML = targets.map(t => {
  const rs = recent.filter(r=>r.target===t), done = rs.filter(r=>r.turnaround_s!=null);
  const last = rs[rs.length-1], open = DATA.open.find(o=>o.target===t);
  const lost = rs.reduce((n,r)=>n+(r.missing||0),0), sent = rs.reduce((n,r)=>n+(r.records||0),0);
  const drift = rs.filter(r=>r.compared_with_previous);
  const avgDrift = drift.length ? drift.reduce((n,r)=>n+r.changed_vs_previous/r.compared_with_previous,0)/drift.length : null;
  return `<div class="tile"><div class="name"><i class="key" style="background:${colorOf(t)}"></i>${esc(label(t))}</div><dl>
    <dt>Median turnaround</dt><dd>${dur(median(done.map(r=>r.turnaround_s)))}</dd>
    <dt>Slowest</dt><dd>${dur(done.length?Math.max(...done.map(r=>r.turnaround_s)):null)}</dd>
    <dt>Lines lost</dt><dd>${sent? `${lost} of ${sent.toLocaleString()}` : "–"}</dd>
    <dt>Answers changed per run</dt><dd>${pct(avgDrift)}</dd>
    <dt>Now</dt><dd>${open ? "running, "+dur(nowTs-open.submitted_ts) : (last ? esc(last.state) : "–")}</dd>
  </dl></div>`; }).join("") || `<div class="empty">No runs yet. The first batches go out at the next scheduled slot.</div>`;

// A small SVG chart helper with a hover tooltip per point.
function chart(id, points, {yTicks, yPos, yLabel, lines}){
  const el = document.getElementById(id);
  if(!points.length){ el.innerHTML = `<div class="empty">No data yet.</div>`; return; }
  const W=1000, H=300, L=64, R=16, T=12, B=34;
  const x0=Math.min(...points.map(p=>p.x)), x1=Math.max(...points.map(p=>p.x)), span=Math.max(x1-x0, DAY);
  const X = x => L + (x-x0)/span*(W-L-R), Y = v => T + (1-yPos(v))*(H-T-B);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yLabel)}">`;
  for(const [v,txt] of yTicks){ const y=Y(v);
    s += `<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="var(--grid)" stroke-width="1"/>`+
         `<text x="${L-8}" y="${y+4}" text-anchor="end">${txt}</text>`; }
  const days = Math.max(1, Math.round(span/DAY)), step = Math.max(1, Math.ceil(days/8));
  for(let d=0; d<=days; d+=step){ const ts = Math.floor(x0/DAY)*DAY + d*DAY; if(ts<x0-1) continue;
    s += `<text x="${X(ts)}" y="${H-10}" text-anchor="middle">${new Date(ts*1000).toISOString().slice(5,10)}</text>`; }
  if(lines){ for(const t of targets){ const ps = points.filter(p=>p.t===t);
    if(ps.length>1) s += `<polyline fill="none" stroke="${colorOf(t)}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="${ps.map(p=>X(p.x)+","+Y(p.y)).join(" ")}"/>`; } }
  points.forEach((p,i)=>{ s += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="4.5" fill="${colorOf(p.t)}" stroke="var(--surface)" stroke-width="2"/>`+
    `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="12" fill="transparent" data-i="${i}"/>`; });
  el.innerHTML = s + `</svg><div class="tip"></div>`;
  const tip = el.querySelector(".tip"), svg = el.querySelector("svg");
  svg.addEventListener("mousemove", e => { const i = e.target.dataset && e.target.dataset.i;
    if(i==null){ tip.style.opacity=0; return; }
    const r = svg.getBoundingClientRect(), p = points[+i];
    tip.innerHTML = p.tip; tip.style.left = (X(p.x)/W*r.width)+"px"; tip.style.top = (Y(p.y)/H*r.height)+"px"; tip.style.opacity=1; });
  svg.addEventListener("mouseleave", () => tip.style.opacity=0);
}

// Turnaround: log scale from 1 minute to 48 hours.
const lmin=Math.log(60), lmax=Math.log(48*3600);
chart("turn", rows.filter(r=>r.turnaround_s!=null).map(r=>({t:r.target, x:r.submitted_ts, y:Math.max(r.turnaround_s,60),
  tip:`${esc(label(r.target))}<br>${when(r.submitted_ts)}<br><b>${dur(r.turnaround_s)}</b>`})),
  {yLabel:"Turnaround per batch", yPos:v=>(Math.log(v)-lmin)/(lmax-lmin),
   yTicks:[[60,"1 min"],[600,"10 min"],[3600,"1 h"],[6*3600,"6 h"],[24*3600,"24 h"]]});
legend("legend1");

// Drift: share of compared answers that changed, 0 to the max seen (at least 10%).
const dpts = rows.filter(r=>r.compared_with_previous).map(r=>({t:r.target, x:r.submitted_ts,
  y:r.changed_vs_previous/r.compared_with_previous,
  tip:`${esc(label(r.target))}<br>${when(r.submitted_ts)}<br><b>${r.changed_vs_previous} of ${r.compared_with_previous}</b> changed`}));
const dmax = Math.max(...dpts.map(p=>p.y), 0), dtop = [0.05,0.1,0.2,0.5,1].find(v=>v>=dmax) || 1;
chart("drift", dpts, {yLabel:"Answers changed since previous run", lines:true, yPos:v=>v/dtop,
  yTicks:[0,.2,.4,.6,.8,1].map(f=>[f*dtop, +(f*dtop*100).toFixed(1)+"%"])});
legend("legend2");

// By submission hour.
const hours = [...new Set(recent.map(r=>r.submit_hour_utc))].sort((a,b)=>a-b);
document.getElementById("slots").innerHTML = recent.length ? `<tr><th>Provider and model</th>${hours.map(h=>
  `<th class="num">${String(h).padStart(2,"0")}:00 median</th><th class="num">${String(h).padStart(2,"0")}:00 slowest</th>`).join("")}</tr>` +
  targets.map(t => `<tr><td><i class="key" style="display:inline-block;margin-right:6px;background:${colorOf(t)}"></i>${esc(label(t))}</td>${hours.map(h=>{
    const v = recent.filter(r=>r.target===t && r.submit_hour_utc===h && r.turnaround_s!=null).map(r=>r.turnaround_s);
    return `<td class="num">${dur(median(v))}</td><td class="num">${dur(v.length?Math.max(...v):null)}</td>`; }).join("")}</tr>`).join("")
  : `<tr><td class="empty">No data yet.</td></tr>`;

// Incidents.
function problems(r){ const p=[];
  if(r.state==="submit_failed") p.push(["bad","submit refused: "+(r.error_message||"")]);
  if(r.state==="failed") p.push(["bad","batch failed ("+(r.provider_detail||"")+")"]);
  if(r.state==="gave_up") p.push(["bad","no result after 50 h"]);
  if(r.missing && r.state!=="gave_up") p.push(["bad",`${r.missing} lines lost`]);
  if(r.error) p.push(["warn",`${r.error} lines errored`]);
  if(r.invalid) p.push(["warn",`${r.invalid} answers not valid JSON for the schema`]);
  if(r.turnaround_s>6*3600) p.push(["warn","took "+dur(r.turnaround_s)]);
  if(r.compared_with_previous && r.changed_vs_previous/r.compared_with_previous>0.05)
    p.push(["warn",`${pct(r.changed_vs_previous/r.compared_with_previous)} of answers changed`]);
  if(r.model_versions && r.model_versions.length>1) p.push(["warn","answered by several model versions: "+r.model_versions.join(", ")]);
  return p; }
const inc = rows.map(r=>[r,problems(r)]).filter(([r,p])=>p.length).reverse();
document.getElementById("incidents").innerHTML = inc.length ? `<tr><th>Sent</th><th>Provider and model</th><th>What happened</th></tr>` +
  inc.map(([r,p])=>`<tr><td>${when(r.submitted_ts)}</td><td>${esc(label(r.target))}</td><td>${p.map(([k,t])=>`<span class="flag ${k}">${esc(t)}</span>`).join("; ")}</td></tr>`).join("")
  : `<tr><td class="empty">None so far.</td></tr>`;

// All runs, newest first.
document.getElementById("all").innerHTML = `<tr><th>Sent</th><th>Provider and model</th><th>State</th><th class="num">Turnaround</th><th class="num">OK</th><th class="num">Lost</th><th class="num">Errored</th><th class="num">Invalid</th><th class="num">Changed</th><th class="num">All 4 fields right</th><th class="num">Cost per 1,000</th><th>Model version</th></tr>` +
  rows.slice().reverse().map(r=>`<tr><td>${when(r.submitted_ts)}</td><td>${esc(label(r.target))}</td><td>${esc(r.state)}</td>
  <td class="num">${dur(r.turnaround_s)}</td><td class="num">${r.ok??"–"}</td><td class="num">${r.missing??"–"}</td>
  <td class="num">${typeof r.error==="number"?r.error:"–"}</td><td class="num">${r.invalid??"–"}</td>
  <td class="num">${r.compared_with_previous?`${r.changed_vs_previous}/${r.compared_with_previous}`:"–"}</td>
  <td class="num">${pct(r.exact_accuracy)}</td><td class="num">${r.cost_per_1000_usd!=null?"$"+r.cost_per_1000_usd.toFixed(3):"–"}</td>
  <td>${esc((r.model_versions||[]).join(", "))}</td></tr>`).join("");
</script>
</body></html>
"""
