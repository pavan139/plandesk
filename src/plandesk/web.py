"""Minimal local web demo: one page, one JSON endpoint."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .answer import answer

app = FastAPI(title="PlanDesk")


@lru_cache(maxsize=1)
def _deps():
    from openai import OpenAI

    from .retrieve import default_retriever

    return default_retriever(), OpenAI()


class Ask(BaseModel):
    question: str
    model: str = "gpt-5.4-mini"


@app.post("/api/ask")
def ask(body: Ask) -> dict:
    retriever, client = _deps()
    return answer(body.question[:500], retriever, client, model=body.model).to_dict()


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PlanDesk</title>
<style>
:root{--bg:#fafaf8;--fg:#1b1b1b;--muted:#666;--card:#fff;--line:#e3e3df;--ok:#1d7a46;--bad:#b3261e;--accent:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--muted:#9a9a9a;--card:#1e1e1e;--line:#333;--ok:#5cc58a;--bad:#ff8a80;--accent:#8aa8ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:32px 16px}
h1{margin:0 0 4px;font-size:28px}p.sub{margin:0 0 24px;color:var(--muted)}
form{display:flex;gap:8px}input{flex:1;padding:12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font:inherit}
button{padding:12px 18px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;cursor:pointer}
.card{margin-top:20px;padding:16px;border:1px solid var(--line);border-radius:10px;background:var(--card)}
.cite{margin-top:10px;padding-left:12px;border-left:3px solid var(--line);font-size:14px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.meta{color:var(--muted);font-size:13px;margin-top:12px}
.ex{margin-top:12px;font-size:14px;color:var(--muted)}.ex a{color:var(--accent);cursor:pointer}
</style></head><body><main>
<h1>PlanDesk</h1>
<p class="sub">Answers about U.S. retirement-plan rules from IRS Pubs 590-A, 590-B and 560 (2024 and 2025), with verified quotes. It says so when the publications don't cover a question. Not tax advice.</p>
<form id="f"><input id="q" placeholder="e.g. What is the SEP IRA limit for 2025?" autocomplete="off"><button>Ask</button></form>
<div class="ex">Try: <a>What is the maximum annual exclusion for QCDs for 2024?</a> · <a>What is the 401(k) limit for 2028?</a> · <a>What is the HSA family limit for 2025?</a></div>
<div id="out"></div>
<script>
const out=document.getElementById('out'),q=document.getElementById('q');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
document.querySelectorAll('.ex a').forEach(a=>a.onclick=()=>{q.value=a.textContent;go()});
document.getElementById('f').onsubmit=e=>{e.preventDefault();go()};
async function go(){if(!q.value.trim())return;out.innerHTML='<div class="card">Searching the publications...</div>';
const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q.value})});
if(!r.ok){out.innerHTML='<div class="card bad">The server could not answer (is OPENAI_API_KEY set?).</div>';return}
const d=await r.json();
let h=`<div class="card"><div>${d.abstained?'<strong>Not answered.</strong> ':''}${esc(d.answer)}</div>`;
for(const c of d.citations){h+=`<div class="cite"><span class="${c.verified?'ok':'bad'}">${c.verified?'Verified quote':'Quote not found in source'}</span> · ${esc(c.source||c.id)}<br>"${esc(c.quote)}"</div>`}
h+=`<div class="meta">${d.model} · ${d.latency_s.toFixed(2)}s · ${d.input_tokens+d.output_tokens} tokens${d.year_filter?` · searched ${d.year_filter} edition`:''}${d.guard?` · guard: ${d.guard}`:''}</div></div>`;
out.innerHTML=h}
</script></main></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE
