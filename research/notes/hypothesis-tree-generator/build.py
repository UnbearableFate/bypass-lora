import json
import os
import runpy
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
DATA_VERSION = sys.argv[2]
DUMP = os.path.join(HERE, "prose-dump.txt")
JS_OUT = os.path.join(HERE, "page-script.js")
JSON_OUT = os.path.join(HERE, "page-data.json")

src = runpy.run_path(os.path.join(HERE, "nodes.py"))
LINES, NODES, GLOSSARY, RUN_STATE = src["LINES"], src["NODES"], src["GLOSSARY"], src["RUN_STATE"]


def rec(i):
    with open(os.path.join(HERE, "rec", i + ".json"), encoding="utf-8") as f:
        return json.load(f)


nodes = {}
for i in range(1, 30):
    for kind in ("H", "E"):
        nid = f"{kind}-{i:04d}"
        d = rec(nid)
        r = d["record"]
        prose = NODES[nid]
        n = {"kind": kind, "title": prose["title"], "purpose": prose["purpose"],
             "did": prose["did"], "result": prose["result"]}
        if kind == "H":
            n["ev"] = r["evidence_status"]
            n["disp"] = r["disposition"]
            n["derived"] = sorted(o["id"] for o in d["out"]
                                  if o["rel"] == "derived_from" and o["id"].startswith("H-"))
        else:
            n["stage"] = r["stage"]
            n["run"] = RUN_STATE.get(nid, "done")
            n["tests"] = sorted(set(o["id"] for o in d["out"] if o["rel"] == "tests"))
        nodes[nid] = n

assert set(nodes) == set(NODES), set(nodes) ^ set(NODES)

generated = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M JST")
data = {"lines": LINES, "nodes": nodes,
        "meta": {"data_version": DATA_VERSION, "generated": generated}}

hs = [n for n in nodes.values() if n["kind"] == "H"]
es = [n for n in nodes.values() if n["kind"] == "E"]
cnt = {k: sum(1 for n in hs if n["ev"] == k) for k in ("supported", "challenged", "inconclusive", "untested")}
cand = sum(1 for n in hs if n["disp"] == "candidate")


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


LEDE = ("29 个假说、29 个实验分属 7 条研究线。得到支持的有 4 个：两项实现检查、M 线的单独编码写入，"
        "以及 RB1-09 改进线的一项筛选（同一 up 接口下，只看当前 token 的 LoRA 没有比历史混合侧支高 1 pp 以上）。"
        "在已完成的普通设置比较中，侧支组的宏平均都低于参数相近、改全部七个模块的 LoRA。")
CAPTION = ("受挑战的 12 个假说有 7 个在 M 线与 RB 线，3 个在 RB1-09 改进线；未检验的 5 个里，"
           "H-0027、H-0028 是 RB1-09 改进线上没有启用的候选，H-0029 是下一步的 A2 确认。")
assert len(hs) == 29 and len(es) == 29
assert cnt == {"supported": 4, "challenged": 12, "inconclusive": 8, "untested": 5}, cnt

gloss_html = "\n".join(
    f"<dt>{esc(t)}</dt><dd>{esc(d)}</dd>" for t, d in GLOSSARY)

json_text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
json_text = json_text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
with open(JSON_OUT, "w", encoding="utf-8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=1))

CSS = r"""
:root{
  --bg:#f7f6f2;--surface:#ffffff;--surface-2:#efede7;--text:#1c1e21;--text-dim:#575c63;
  --border:#e0ddd4;--border-bright:#8a8f97;--accent:#0b6680;--accent-bg:#e2eff3;
  --ok:#2c7548;--ok-bg:#e3f1e8;--risk:#b42318;--risk-bg:#fbe8e5;--warn:#965800;--warn-bg:#faefda;
  --neutral:#686d75;--neutral-bg:#eeedea;
  --font-sans:"Source Han Sans SC","Noto Sans CJK SC","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Helvetica Neue","Segoe UI",sans-serif;
  --font-serif:"Source Han Serif SC","Noto Serif CJK SC","Songti SC","STSong","Noto Serif",Georgia,serif;
  --font-mono:"JetBrains Mono","SFMono-Regular","Menlo","Consolas","Liberation Mono",monospace;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#121417;--surface:#1a1d21;--surface-2:#23272c;--text:#e8e6e1;--text-dim:#a5aab1;
    --border:#2d3137;--border-bright:#6f757e;--accent:#62bad4;--accent-bg:#15313a;
    --ok:#64c690;--ok-bg:#162e20;--risk:#f28b7f;--risk-bg:#3b1d1a;--warn:#e5ab42;--warn-bg:#382a12;
    --neutral:#9ba1aa;--neutral-bg:#262a30;color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#121417;--surface:#1a1d21;--surface-2:#23272c;--text:#e8e6e1;--text-dim:#a5aab1;
  --border:#2d3137;--border-bright:#6f757e;--accent:#62bad4;--accent-bg:#15313a;
  --ok:#64c690;--ok-bg:#162e20;--risk:#f28b7f;--risk-bg:#3b1d1a;--warn:#e5ab42;--warn-bg:#382a12;
  --neutral:#9ba1aa;--neutral-bg:#262a30;color-scheme:dark;
}
*{box-sizing:border-box}
[hidden]{display:none !important}
.sr{position:absolute;width:1px;height:1px;margin:-1px;padding:0;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.7 var(--font-sans)}
button,input{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.wrap{max-width:1200px;margin:0 auto;padding:0 16px}
.mono{font-family:var(--font-mono)}
.top{padding:28px 0 4px}
.kicker{font:12px/1.4 var(--font-mono);color:var(--text-dim);margin:0 0 6px}
h1{font-family:var(--font-serif);font-weight:700;font-size:30px;line-height:1.25;margin:0 0 10px}
.lede{max-width:62ch;margin:0;font-size:16px}
.overview{margin:20px 0 28px;padding:16px;background:var(--surface);border:1px solid var(--border);border-radius:8px}
.ov-legend{display:flex;flex-wrap:wrap;gap:4px 16px;list-style:none;margin:0 0 8px;padding:0;font-size:13px;color:var(--text-dim)}
.ov-legend li{display:inline-flex;align-items:center;gap:6px}
.ov-legend b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums}
.ov-row{display:grid;grid-template-columns:minmax(0,15em) minmax(0,1fr);gap:4px 14px;align-items:start;padding:8px 0;border-top:1px solid var(--border)}
.ov-name{font-size:13px;color:var(--text-dim);padding-top:2px}
.ov-chips{display:flex;flex-wrap:wrap;gap:6px}
.ov-chip{display:inline-flex;align-items:center;gap:5px;padding:1px 7px 1px 4px;border:1px solid var(--border);border-radius:4px;background:var(--surface);font:12px/1.6 var(--font-mono);cursor:pointer;font-variant-numeric:tabular-nums}
.ov-chip:hover{border-color:var(--border-bright)}
.ov-chip[aria-current="true"]{border-color:var(--accent);background:var(--accent-bg)}
.sw{display:inline-block;width:14px;height:14px;border:1.5px solid;border-radius:2px;flex:none}
.sw.supported{background:var(--ok);border-color:var(--ok)}
.sw.challenged{border-color:var(--risk);background:repeating-linear-gradient(135deg,var(--risk) 0 2px,transparent 2px 5px)}
.sw.inconclusive{border-color:var(--warn);background:linear-gradient(90deg,var(--warn) 50%,transparent 50%)}
.sw.untested{border-color:var(--neutral);background:transparent}
.sw.cand{border-style:dashed}
.overview figcaption{margin-top:10px;font-size:14px;max-width:65ch}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 10px}
.btn{padding:4px 12px;border:1px solid var(--border-bright);border-radius:4px;background:var(--surface);cursor:pointer;font-size:14px}
.btn:hover{background:var(--surface-2)}
.filter{display:flex;align-items:center;gap:6px;flex:1 1 230px;min-width:0;font-size:14px;color:var(--text-dim)}
.filter input{flex:1;min-width:0;padding:4px 8px;border:1px solid var(--border-bright);border-radius:4px;background:var(--surface);color:var(--text);font-size:14px}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:12.5px;color:var(--text-dim);margin:0 0 16px;padding:8px 0;border-top:1px solid var(--border);border-bottom:1px solid var(--border)}
.legend .grp{display:flex;flex-wrap:wrap;gap:4px 6px;align-items:center}
.legend .grp-name{margin-right:2px}
.legend .en{font-family:var(--font-mono);font-size:11px;color:var(--text-dim)}
.cols{display:grid;grid-template-columns:minmax(0,1fr);gap:20px;align-items:start}
.tree,.kids{list-style:none;margin:0;padding:0}
.kids{padding-left:14px;margin-left:11px;border-left:1px solid var(--border)}
.kids.top{margin-left:11px}
.line{margin:0 0 14px}
.line-tog{display:flex;align-items:baseline;flex-wrap:wrap;gap:2px 10px;width:100%;text-align:left;padding:4px 2px;border:0;background:transparent;cursor:pointer}
.line-tog svg{align-self:center;flex:none;color:var(--text-dim);transition:transform .15s}
.line-tog[aria-expanded="true"] svg{transform:rotate(90deg)}
.lname{font-weight:700;font-size:16px}
.count{font-size:12.5px;color:var(--text-dim);font-variant-numeric:tabular-nums}
.line-desc{margin:0 0 6px 22px;font-size:13px;color:var(--text-dim);max-width:65ch}
.row{display:flex;align-items:flex-start;gap:2px}
.tog{flex:none;width:24px;height:30px;padding:0;border:0;background:transparent;cursor:pointer;color:var(--text-dim);display:grid;place-items:center;border-radius:4px}
.tog:hover{background:var(--surface-2)}
.tog svg{transition:transform .15s}
.tog[aria-expanded="true"] svg{transform:rotate(90deg)}
.spacer{flex:none;width:24px;height:30px;display:grid;place-items:center}
.spacer::before{content:"";width:5px;height:5px;border-radius:50%;background:var(--border-bright)}
.sel{flex:1;min-width:0;display:flex;flex-wrap:wrap;align-items:baseline;gap:2px 8px;text-align:left;padding:3px 6px;border:1px solid transparent;border-radius:4px;background:transparent;cursor:pointer}
.sel:hover{background:var(--surface-2)}
.sel[aria-current="true"]{background:var(--accent-bg);border-color:var(--accent)}
.nid{font:12.5px/1.7 var(--font-mono);color:var(--text-dim);font-variant-numeric:tabular-nums}
.node.h>.row .nid{color:var(--text);font-weight:600}
.ttl{flex:1 1 12em;min-width:0;overflow-wrap:anywhere}
.node.e>.row .ttl{color:var(--text-dim)}
.badges{display:inline-flex;flex-wrap:wrap;gap:4px}
.b{display:inline-flex;align-items:center;gap:3px;padding:0 6px;border:1px solid var(--border-bright);border-radius:4px;font-size:12px;line-height:20px;white-space:nowrap;background:var(--surface);color:var(--text)}
.b .g{font-size:11px;line-height:1}
.b.ev-supported{border-color:var(--ok);background:var(--ok-bg)}.b.ev-supported .g{color:var(--ok)}
.b.ev-challenged{border-color:var(--risk);background:var(--risk-bg)}.b.ev-challenged .g{color:var(--risk)}
.b.ev-inconclusive{border-color:var(--warn);background:var(--warn-bg)}.b.ev-inconclusive .g{color:var(--warn)}
.b.ev-untested{border-color:var(--neutral);background:var(--neutral-bg)}.b.ev-untested .g{color:var(--neutral)}
.b.disp{border-color:var(--border);background:transparent;color:var(--text-dim)}
.b.disp-active .g{color:var(--accent)}
.b.disp-candidate{border-style:dashed}
.b.stage{border-color:var(--border);background:transparent;color:var(--text-dim)}
.b.stage .mono{color:var(--text);font-size:11.5px}
.b.run-done{border-color:var(--border);background:transparent;color:var(--text-dim)}
.b.run-running{border-color:var(--accent);background:var(--accent-bg)}.b.run-running .g{color:var(--accent)}
.b.run-notstarted{border-style:dashed;border-color:var(--border-bright);background:transparent;color:var(--text-dim)}
.detail{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px 18px}
.detail .hint{margin:0;color:var(--text-dim);font-size:14px}
.d-id{font:12.5px/1.5 var(--font-mono);color:var(--text-dim);margin:0 0 4px;overflow-wrap:anywhere}
.d-title{font-family:var(--font-serif);font-size:21px;line-height:1.4;margin:0 0 8px}
.d-title:focus{outline:none}
.d-title:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.d-badges{display:flex;flex-wrap:wrap;gap:4px}
.d-rel{font-size:13px;color:var(--text-dim);margin:8px 0 0}
.d-rel button{font:12.5px/1.6 var(--font-mono);border:1px solid var(--border-bright);border-radius:4px;background:transparent;padding:0 6px;cursor:pointer;margin:0 4px 2px 0}
.d-rel button:hover{background:var(--surface-2)}
.detail section{margin-top:18px}
.detail h3{font-size:13.5px;letter-spacing:.06em;color:var(--accent);margin:0 0 4px;font-weight:700}
.detail section p{margin:0;max-width:65ch;overflow-wrap:anywhere}
.back{margin-top:18px}
.no-match{color:var(--text-dim);font-size:14px}
.glossary{margin:36px 0 0;border-top:1px solid var(--border);padding-top:12px}
.glossary summary{cursor:pointer;font-weight:700;font-size:16px}
.glossary dl{display:grid;grid-template-columns:minmax(0,1fr);gap:2px 18px;margin:14px 0 0}
.glossary dt{font-weight:600;overflow-wrap:anywhere}
.glossary dd{margin:0 0 10px;max-width:65ch;overflow-wrap:anywhere}
.foot{padding-top:24px;padding-bottom:40px;color:var(--text-dim);font-size:13px}
.foot p{margin:0;max-width:80ch}
@media (min-width:760px){.glossary dl{grid-template-columns:minmax(0,15em) minmax(0,1fr)}}
@media (max-width:640px){.ov-row{grid-template-columns:minmax(0,1fr)}h1{font-size:26px}}
@media (min-width:900px){
  .cols{grid-template-columns:minmax(0,1.15fr) minmax(0,1fr)}
  .detail{position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto}
  .back{display:none}
}
@media (prefers-reduced-motion:reduce){*{transition:none !important;scroll-behavior:auto !important}}
"""

JS = r"""
var HT = (function () {
  'use strict';
  var EV = {supported: ['■', '支持'], challenged: ['▨', '受挑战'], inconclusive: ['◧', '无定论'], untested: ['□', '未检验']};
  var DISP = {active: ['●', '活跃'], paused: ['‖', '暂停'], retired: ['×', '已关闭'], candidate: ['◌', '候选']};
  var STAGE = {E0: '实现', E1: '机制', E2: '真实模型', E3: '确认'};
  var RUN = {done: ['✓', '已完成'], running: ['◐', '进行中'], notstarted: ['○', '未开始']};
  var CARET = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true" focusable="false"><path d="M4 2.5 8 6 4 9.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  var ESC = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
  function esc(s) { return String(s).replace(/[&<>"']/g, function (c) { return ESC[c]; }); }
  function badge(cls, glyph, word, title) {
    return '<span class="b ' + cls + '" title="' + esc(title) + '"><span class="g" aria-hidden="true">' + glyph + '</span>' + esc(word) + '</span>';
  }
  function badgesHTML(n) {
    if (n.kind === 'H') {
      return badge('ev ev-' + n.ev, EV[n.ev][0], EV[n.ev][1], '证据状态：' + n.ev) +
        badge('disp disp-' + n.disp, DISP[n.disp][0], DISP[n.disp][1], '处置：' + n.disp);
    }
    return '<span class="b stage" title="实验阶段"><span class="mono">' + esc(n.stage) + '</span>' + esc(STAGE[n.stage]) + '</span>' +
      badge('run run-' + n.run, RUN[n.run][0], RUN[n.run][1], '进度');
  }
  function walk(list, fn) { list.forEach(function (it) { fn(it[0]); walk(it[1], fn); }); }
  function lineIds(line) {
    var seen = {}, out = [];
    walk(line.tree, function (id) { if (!seen[id]) { seen[id] = 1; out.push(id); } });
    return out;
  }
  function countLine(line) {
    var ids = lineIds(line), h = 0, e = 0;
    ids.forEach(function (id) { if (id.charAt(0) === 'H') { h++; } else { e++; } });
    return [h, e];
  }
  function nodeHTML(data, item, path) {
    var id = item[0], kids = item[1], n = data.nodes[id], gid = 'g-' + path;
    var s = '<li class="node ' + (n.kind === 'H' ? 'h' : 'e') + '" data-id="' + id + '" id="r-' + path + '"><div class="row">';
    if (kids.length) {
      s += '<button type="button" class="tog" aria-expanded="false" aria-controls="' + gid + '" aria-label="展开或折叠 ' + id + '">' + CARET + '</button>';
    } else {
      s += '<span class="spacer" aria-hidden="true"></span>';
    }
    s += '<button type="button" class="sel" data-id="' + id + '"><span class="nid">' + id + '</span><span class="ttl">' + esc(n.title) +
      '</span><span class="badges">' + badgesHTML(n) + '</span></button></div>';
    if (kids.length) {
      s += '<ul class="kids" id="' + gid + '" hidden>';
      kids.forEach(function (k, i) { s += nodeHTML(data, k, path + '-' + i); });
      s += '</ul>';
    }
    return s + '</li>';
  }
  function treeHTML(data) {
    return data.lines.map(function (line) {
      var c = countLine(line), gid = 'g-' + line.key;
      var s = '<li class="line" data-line="' + line.key + '"><button type="button" class="line-tog" aria-expanded="true" aria-controls="' + gid + '">' +
        CARET + '<span class="lname">' + esc(line.name) + '</span><span class="count">' + c[0] + ' 个假说 · ' + c[1] + ' 个实验</span></button>' +
        '<p class="line-desc">' + esc(line.desc) + '</p><ul class="kids top" id="' + gid + '">';
      line.tree.forEach(function (it, i) { s += nodeHTML(data, it, line.key + '-' + i); });
      return s + '</ul></li>';
    }).join('');
  }
  function overviewHTML(data) {
    return data.lines.map(function (line) {
      var chips = lineIds(line).filter(function (id) { return id.charAt(0) === 'H'; }).map(function (id) {
        var n = data.nodes[id];
        var label = id + ' ' + EV[n.ev][1] + '，' + DISP[n.disp][1] + '：' + n.title;
        return '<button type="button" class="ov-chip" data-id="' + id + '" aria-label="' + esc(label) + '" title="' + esc(label) + '">' +
          '<span class="sw ' + n.ev + (n.disp === 'candidate' ? ' cand' : '') + '" aria-hidden="true"></span>' + id + '</button>';
      }).join('');
      return '<div class="ov-row"><div class="ov-name">' + esc(line.name) + '</div><div class="ov-chips">' + chips + '</div></div>';
    }).join('');
  }
  function lineNames(data) {
    var m = {};
    data.lines.forEach(function (line) { lineIds(line).forEach(function (id) { m[id] = line.name; }); });
    return m;
  }
  return {esc: esc, badgesHTML: badgesHTML, treeHTML: treeHTML, overviewHTML: overviewHTML,
    countLine: countLine, lineIds: lineIds, lineNames: lineNames, EV: EV, DISP: DISP, STAGE: STAGE, RUN: RUN};
})();

function htInit() {
  'use strict';
  var data = JSON.parse(document.getElementById('tree-data').textContent);
  var tree = document.getElementById('tree');
  var ov = document.getElementById('ov-grid');
  var card = document.getElementById('detail-card');
  var empty = document.getElementById('detail-empty');
  var dId = document.getElementById('d-id');
  var dTitle = document.getElementById('d-title');
  var dBadges = document.getElementById('d-badges');
  var dRel = document.getElementById('d-rel');
  var noMatch = document.getElementById('no-match');
  var filter = document.getElementById('filter');
  var names = HT.lineNames(data);
  var mqNarrow = window.matchMedia('(max-width: 899px)');
  var mqReduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  var lastRow = null, saved = null;
  tree.innerHTML = HT.treeHTML(data);
  ov.innerHTML = HT.overviewHTML(data);

  function behavior() { return mqReduce.matches ? 'auto' : 'smooth'; }
  function setExpanded(btn, open) {
    if (!btn) { return; }
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    var g = document.getElementById(btn.getAttribute('aria-controls'));
    if (g) { g.hidden = !open; }
  }
  function toggles() { return tree.querySelectorAll('.tog, .line-tog'); }
  function childList(li) {
    for (var c = li.firstElementChild; c; c = c.nextElementSibling) { if (c.tagName === 'UL') { return c; } }
    return null;
  }
  function ownToggle(li) {
    if (li.classList.contains('line')) { return li.firstElementChild; }
    var row = li.firstElementChild, b = row && row.firstElementChild;
    return b && b.classList.contains('tog') ? b : null;
  }
  function reveal(li) {
    var p = li.parentElement;
    while (p && p !== tree) {
      if (p.tagName === 'UL' && p.id) { setExpanded(tree.querySelector('[aria-controls="' + p.id + '"]'), true); }
      p = p.parentElement;
    }
  }
  function rowFor(id) { return tree.querySelector('li[data-id="' + id + '"]'); }
  function relButtons(label, ids) {
    var frag = document.createDocumentFragment();
    frag.appendChild(document.createTextNode(label));
    ids.forEach(function (rid) {
      var b = document.createElement('button');
      b.type = 'button';
      b.textContent = rid;
      b.setAttribute('data-rel', rid);
      frag.appendChild(b);
    });
    return frag;
  }
  function select(id, opt) {
    var n = data.nodes[id];
    if (!n) { return; }
    opt = opt || {};
    dId.textContent = id + ' · ' + (n.kind === 'H' ? '假说' : '实验') + ' · ' + (names[id] || '');
    dTitle.textContent = n.title;
    dBadges.innerHTML = HT.badgesHTML(n);
    dRel.textContent = '';
    if (n.kind === 'H' && n.derived.length) { dRel.appendChild(relButtons('源自：', n.derived)); }
    if (n.kind === 'E' && n.tests.length) { dRel.appendChild(relButtons('检验：', n.tests)); }
    dRel.hidden = !dRel.firstChild;
    document.getElementById('d-purpose').textContent = n.purpose;
    document.getElementById('d-did').textContent = n.did;
    document.getElementById('d-result').textContent = n.result;
    empty.hidden = true;
    card.hidden = false;
    tree.querySelectorAll('.sel[aria-current]').forEach(function (b) { b.removeAttribute('aria-current'); });
    tree.querySelectorAll('.sel[data-id="' + id + '"]').forEach(function (b) { b.setAttribute('aria-current', 'true'); });
    ov.querySelectorAll('.ov-chip[aria-current]').forEach(function (b) { b.removeAttribute('aria-current'); });
    var chip = ov.querySelector('.ov-chip[data-id="' + id + '"]');
    if (chip) { chip.setAttribute('aria-current', 'true'); }
    var row = opt.row || rowFor(id);
    if (opt.reveal && row) { reveal(row); }
    lastRow = row;
    document.getElementById('sr-status').textContent = '已显示 ' + id + ' 的详情';
    try { history.replaceState(null, '', '#' + id); } catch (e) { }
    if (!opt.from) { return; }
    if (mqNarrow.matches) {
      card.scrollIntoView({block: 'start', behavior: behavior()});
      dTitle.focus({preventScroll: true});
    } else if (opt.from !== 'tree' && row && !row.hidden) {
      row.scrollIntoView({block: 'center', behavior: behavior()});
    }
  }

  tree.addEventListener('click', function (ev) {
    var b = ev.target.closest('button');
    if (!b || !tree.contains(b)) { return; }
    if (b.classList.contains('tog') || b.classList.contains('line-tog')) {
      setExpanded(b, b.getAttribute('aria-expanded') !== 'true');
    } else if (b.classList.contains('sel')) {
      select(b.getAttribute('data-id'), {from: 'tree', row: b.closest('li')});
    }
  });
  ov.addEventListener('click', function (ev) {
    var b = ev.target.closest('.ov-chip');
    if (b) { select(b.getAttribute('data-id'), {from: 'overview', reveal: true}); }
  });
  dRel.addEventListener('click', function (ev) {
    var b = ev.target.closest('button[data-rel]');
    if (b) { select(b.getAttribute('data-rel'), {from: 'rel', reveal: true}); }
  });
  document.getElementById('back-to-tree').addEventListener('click', function () {
    if (!lastRow) { return; }
    reveal(lastRow);
    lastRow.scrollIntoView({block: 'center', behavior: behavior()});
    var s = lastRow.querySelector('.sel');
    if (s) { s.focus({preventScroll: true}); }
  });
  document.getElementById('expand-all').addEventListener('click', function () {
    toggles().forEach(function (b) { setExpanded(b, true); });
  });
  document.getElementById('collapse-all').addEventListener('click', function () {
    toggles().forEach(function (b) { setExpanded(b, false); });
  });

  function matchNode(li, q) {
    var n = data.nodes[li.getAttribute('data-id')];
    var self = (li.getAttribute('data-id') + ' ' + n.title).toLowerCase().indexOf(q) >= 0;
    var any = false, ul = childList(li);
    if (ul) {
      for (var k = ul.firstElementChild; k; k = k.nextElementSibling) { if (matchNode(k, q)) { any = true; } }
      setExpanded(ownToggle(li), any);
    }
    li.hidden = !(self || any);
    if (self) { li.setAttribute('data-hit', ''); } else { li.removeAttribute('data-hit'); }
    return self || any;
  }
  function applyFilter() {
    var q = filter.value.trim().toLowerCase();
    if (!q) {
      tree.querySelectorAll('li').forEach(function (li) { li.hidden = false; li.removeAttribute('data-hit'); });
      if (saved) { saved.forEach(function (st) { setExpanded(st[0], st[1]); }); saved = null; }
      noMatch.hidden = true;
      return;
    }
    if (!saved) {
      saved = [];
      toggles().forEach(function (b) { saved.push([b, b.getAttribute('aria-expanded') === 'true']); });
    }
    var hits = 0;
    for (var line = tree.firstElementChild; line; line = line.nextElementSibling) {
      var ul = childList(line), any = false;
      if (line.querySelector('.lname').textContent.toLowerCase().indexOf(q) >= 0) {
        line.querySelectorAll('li').forEach(function (li) { li.hidden = false; });
        any = true;
      } else {
        for (var k = ul.firstElementChild; k; k = k.nextElementSibling) { if (matchNode(k, q)) { any = true; } }
      }
      line.hidden = !any;
      setExpanded(ownToggle(line), any);
      if (any) { hits++; }
    }
    noMatch.hidden = hits > 0;
  }
  filter.addEventListener('input', applyFilter);

  var themeBtn = document.getElementById('theme');
  var modes = ['auto', 'light', 'dark'], modeName = {auto: '自动', light: '浅色', dark: '深色'};
  function curMode() {
    var t = document.documentElement.getAttribute('data-theme');
    return t === 'light' || t === 'dark' ? t : 'auto';
  }
  function showMode() { themeBtn.textContent = '配色：' + modeName[curMode()]; }
  themeBtn.addEventListener('click', function () {
    var m = modes[(modes.indexOf(curMode()) + 1) % modes.length];
    if (m === 'auto') { document.documentElement.removeAttribute('data-theme'); } else { document.documentElement.setAttribute('data-theme', m); }
    try { if (m === 'auto') { localStorage.removeItem('ht-theme'); } else { localStorage.setItem('ht-theme', m); } } catch (e) { }
    showMode();
  });
  showMode();

  var h = decodeURIComponent((location.hash || '').slice(1));
  if (data.nodes[h]) { select(h, {reveal: true}); }
}
if (typeof document !== 'undefined' && document.getElementById) {
  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', htInit); } else { htInit(); }
}
"""

with open(JS_OUT, "w", encoding="utf-8") as f:
    f.write(JS)

legend_ev = "".join(
    f'<span class="b ev ev-{k}"><span class="g" aria-hidden="true">{g}</span>{w}</span><span class="en">{k}</span>'
    for k, g, w in (("supported", "■", "支持"), ("challenged", "▨", "受挑战"),
                    ("inconclusive", "◧", "无定论"), ("untested", "□", "未检验")))
legend_disp = "".join(
    f'<span class="b disp disp-{k}"><span class="g" aria-hidden="true">{g}</span>{w}</span><span class="en">{k}</span>'
    for k, g, w in (("active", "●", "活跃"), ("paused", "‖", "暂停"),
                    ("retired", "×", "已关闭"), ("candidate", "◌", "候选")))
legend_stage = "".join(
    f'<span class="b stage"><span class="mono">{k}</span>{w}</span>'
    for k, w in (("E0", "实现"), ("E1", "机制"), ("E2", "真实模型"), ("E3", "确认")))
legend_run = "".join(
    f'<span class="b run run-{k}"><span class="g" aria-hidden="true">{g}</span>{w}</span>'
    for k, g, w in (("done", "✓", "已完成"), ("running", "◐", "进行中"), ("notstarted", "○", "未开始")))

ov_legend = "".join(
    f'<li><span class="sw {k}" aria-hidden="true"></span>{w} <b>{cnt[k]}</b></li>'
    for k, w in (("supported", "支持"), ("challenged", "受挑战"), ("inconclusive", "无定论"), ("untested", "未检验")))
ov_legend += f'<li><span class="sw untested cand" aria-hidden="true"></span>虚线框为候选假说 <b>{cand}</b></li>'

HTML = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>假说探索树</title>
<meta name="description" content="bypass-lora 项目全部假说与实验，按研究线组织成可折叠的树；点击条目查看目的、做法与结果。">
<script>try{{var t=localStorage.getItem('ht-theme');if(t==='light'||t==='dark'){{document.documentElement.setAttribute('data-theme',t);}}}}catch(e){{}}</script>
<style>{CSS}</style>
</head>
<body>
<header class="top">
<div class="wrap">
<p class="kicker">bypass-lora · smooth-research 研究记录</p>
<h1>假说探索树</h1>
<p class="lede">{esc(LEDE)}</p>
</div>
</header>
<main class="wrap">
<figure class="overview" aria-labelledby="ov-cap">
<ul class="ov-legend" aria-label="证据状态计数">{ov_legend}</ul>
<div id="ov-grid"></div>
<figcaption id="ov-cap">{esc(CAPTION)}点击格子查看该假说的详情。</figcaption>
</figure>
<section aria-label="假说与实验">
<div class="toolbar">
<button type="button" class="btn" id="expand-all">全部展开</button>
<button type="button" class="btn" id="collapse-all">全部折叠</button>
<label class="filter"><span>筛选</span><input id="filter" type="search" placeholder="编号或标题，如 H-0012、LoRA" autocomplete="off"></label>
<button type="button" class="btn" id="theme">配色：自动</button>
</div>
<div class="legend" aria-label="图例">
<div class="grp"><span class="grp-name">证据状态</span>{legend_ev}</div>
<div class="grp"><span class="grp-name">处置</span>{legend_disp}</div>
<div class="grp"><span class="grp-name">实验阶段</span>{legend_stage}</div>
<div class="grp"><span class="grp-name">实验进度</span>{legend_run}</div>
</div>
<div class="cols">
<nav aria-label="假说探索树">
<ul class="tree" id="tree"></ul>
<p class="no-match" id="no-match" hidden>没有匹配的条目。</p>
</nav>
<aside class="detail" id="detail" aria-label="详情">
<p class="sr" id="sr-status" aria-live="polite"></p>
<p class="hint" id="detail-empty">点击左侧任一条目，这里显示它的目的、做了什么与结果。窄屏时详情显示在树的下方。</p>
<article id="detail-card" hidden>
<header>
<p class="d-id" id="d-id"></p>
<h2 class="d-title" id="d-title" tabindex="-1"></h2>
<div class="d-badges" id="d-badges"></div>
<p class="d-rel" id="d-rel"></p>
</header>
<section><h3>目的</h3><p id="d-purpose"></p></section>
<section><h3>做了什么</h3><p id="d-did"></p></section>
<section><h3>结果</h3><p id="d-result"></p></section>
<button type="button" class="btn back" id="back-to-tree">回到树</button>
</article>
</aside>
</div>
</section>
<details class="glossary">
<summary>术语表</summary>
<dl>
{gloss_html}
</dl>
</details>
</main>
<footer class="wrap foot">
<p>数据来自本项目的 smooth-research 记录（rl graph、rl show 与 research/experiments 下的实验契约）。数据版本 data_version {esc(DATA_VERSION)}（rl status），生成于 {generated}。</p>
</footer>
<script type="application/json" id="tree-data">{json_text}</script>
<script>{JS}</script>
</body>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)

# Prose dump: every visible Chinese string, one per line.
lines_out = [LEDE, CAPTION + "点击格子查看该假说的详情。"]
for line in LINES:
    lines_out += [line["name"], line["desc"]]
for nid in sorted(nodes):
    n = nodes[nid]
    lines_out += [n["title"], n["purpose"], n["did"], n["result"]]
for t, d in GLOSSARY:
    lines_out += [t, d]
with open(DUMP, "w", encoding="utf-8") as f:
    f.write("\n".join(lines_out) + "\n")
print("wrote", OUT, os.path.getsize(OUT), "bytes; generated", generated)
