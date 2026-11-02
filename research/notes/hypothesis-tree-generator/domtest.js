'use strict';
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const page = fs.readFileSync(process.argv[2], 'utf8');
const dir = process.argv[3];
const data = JSON.parse(fs.readFileSync(dir + '/page-data-roundtrip.json', 'utf8'));
const mainJs = fs.readFileSync(dir + '/page-main.js', 'utf8');

class N { constructor(t) { this.nodeType = t; this.childNodes = []; this.parentNode = null; }
  appendChild(c) {
    if (c.nodeType === 11) { c.childNodes.slice().forEach(x => this.appendChild(x)); c.childNodes = []; return c; }
    if (c.parentNode) c.parentNode.removeChild(c);
    c.parentNode = this; this.childNodes.push(c); return c;
  }
  removeChild(c) { const i = this.childNodes.indexOf(c); if (i >= 0) this.childNodes.splice(i, 1); c.parentNode = null; return c; }
  get firstChild() { return this.childNodes[0] || null; }
  get textContent() { return this.childNodes.map(c => c.textContent).join(''); }
  set textContent(v) { this.childNodes.forEach(c => { c.parentNode = null; }); this.childNodes = []; if (v !== '' && v != null) this.appendChild(new T(String(v))); }
}
class T extends N { constructor(s) { super(3); this.data = s; } get textContent() { return this.data; } set textContent(v) { this.data = String(v); } }
class F extends N { constructor() { super(11); } }
const ENT = {'&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'"};
const dec = s => s.replace(/&(amp|lt|gt|quot|#39);/g, m => ENT[m]);
class E extends N {
  constructor(tag) { super(1); this.tagName = tag.toUpperCase(); this.attrs = {}; this.listeners = {}; }
  getAttribute(n) { return Object.prototype.hasOwnProperty.call(this.attrs, n) ? this.attrs[n] : null; }
  setAttribute(n, v) { this.attrs[n] = String(v); }
  removeAttribute(n) { delete this.attrs[n]; }
  hasAttribute(n) { return Object.prototype.hasOwnProperty.call(this.attrs, n); }
  get id() { return this.getAttribute('id') || ''; }
  get hidden() { return this.hasAttribute('hidden'); }
  set hidden(v) { if (v) this.setAttribute('hidden', ''); else this.removeAttribute('hidden'); }
  set type(v) { this.setAttribute('type', v); }
  get value() { return this.attrs.value || ''; }
  set value(v) { this.attrs.value = String(v); }
  get classList() { const s = this; return {contains: c => (s.getAttribute('class') || '').split(/\s+/).includes(c)}; }
  get children() { return this.childNodes.filter(c => c.nodeType === 1); }
  get firstElementChild() { return this.children[0] || null; }
  get nextElementSibling() { if (!this.parentNode) return null; const s = this.parentNode.children; return s[s.indexOf(this) + 1] || null; }
  get parentElement() { return this.parentNode && this.parentNode.nodeType === 1 ? this.parentNode : null; }
  set innerHTML(h) { this.childNodes.forEach(c => { c.parentNode = null; }); this.childNodes = []; parseInto(this, h); }
  matches(sel) { return sel.split(',').some(s => matchOne(this, s.trim())); }
  closest(sel) { let e = this; while (e && e.nodeType === 1) { if (e.matches(sel)) return e; e = e.parentNode; } return null; }
  contains(n) { while (n) { if (n === this) return true; n = n.parentNode; } return false; }
  querySelectorAll(sel) { const out = []; (function w(e) { e.children.forEach(c => { if (c.matches(sel)) out.push(c); w(c); }); })(this); return out; }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  scrollIntoView() { this.scrolled = (this.scrolled || 0) + 1; }
  focus() { doc.activeElement = this; }
}
function matchOne(el, s) {
  const m = /^([a-zA-Z][\w-]*)?((?:\.[\w-]+|\[[^\]]+\])*)$/.exec(s);
  if (!m) throw new Error('selector unsupported: ' + s);
  if (m[1] && el.tagName !== m[1].toUpperCase()) return false;
  const parts = m[2].match(/\.[\w-]+|\[[^\]]+\]/g) || [];
  return parts.every(p => {
    if (p[0] === '.') return el.classList.contains(p.slice(1));
    const a = /^\[([\w-]+)(?:="([^"]*)")?\]$/.exec(p);
    if (!a) throw new Error('attr selector unsupported: ' + p);
    return a[2] === undefined ? el.hasAttribute(a[1]) : el.getAttribute(a[1]) === a[2];
  });
}
const VOID = new Set(['input', 'meta', 'br', 'img', 'link', 'hr']);
function parseInto(root, html) {
  const re = /<(\/?)([a-zA-Z][\w-]*)((?:\s+[\w:-]+(?:="[^"]*")?)*)\s*(\/?)>|([^<]+)/g;
  let cur = root, m;
  while ((m = re.exec(html))) {
    if (m[5] !== undefined) { cur.appendChild(new T(dec(m[5]))); continue; }
    const tag = m[2].toLowerCase();
    if (m[1]) {
      let e = cur; while (e !== root && e.tagName !== tag.toUpperCase()) e = e.parentNode;
      if (e === root) throw new Error('stray close ' + tag);
      cur = e.parentNode; continue;
    }
    const el = new E(tag);
    const ar = /([\w:-]+)(?:="([^"]*)")?/g; let a;
    while ((a = ar.exec(m[3]))) el.setAttribute(a[1], a[2] === undefined ? '' : dec(a[2]));
    cur.appendChild(el);
    if (!VOID.has(tag) && !m[4]) cur = el;
  }
}
const body = /<body>([\s\S]*)<\/body>/.exec(page)[1].replace(/<script[\s\S]*?<\/script>/g, '').replace(/<style[\s\S]*?<\/style>/g, '');
const root = new E('body'); parseInto(root, body);
const html = new E('html');
const doc = {
  documentElement: html, readyState: 'complete', activeElement: null,
  getElementById: id => root.querySelector('[id="' + id + '"]'),
  createElement: t => new E(t), createTextNode: s => new T(s), createDocumentFragment: () => new F(),
  addEventListener() {}
};
root.querySelector = root.querySelector.bind(root);
let narrow = false, lastUrl = null; const store = {};
const ctx = {
  document: doc, location: {hash: '#H-0012'},
  history: {replaceState: (a, b, u) => { lastUrl = u; }},
  localStorage: {getItem: k => store[k] || null, setItem: (k, v) => { store[k] = v; }, removeItem: k => { delete store[k]; }},
  window: {matchMedia: q => ({get matches() { return q.indexOf('max-width') >= 0 ? narrow : false; }})},
  decodeURIComponent, JSON, String, Object, Array
};
vm.createContext(ctx);
const tdata = doc.getElementById('tree-data');
assert.strictEqual(tdata, null);
const holder = new E('script'); holder.setAttribute('id', 'tree-data'); holder.appendChild(new T(JSON.stringify(data))); root.appendChild(holder);
vm.runInContext(mainJs, ctx);

function click(el) {
  assert.ok(el, 'click target missing');
  const ev = {target: el, type: 'click'};
  for (let e = el; e; e = e.parentNode) (e.listeners && e.listeners.click || []).forEach(f => f(ev));
}
function input(el) { (el.listeners.input || []).forEach(f => f({target: el})); }
const $ = id => doc.getElementById(id);
const tree = $('tree'), card = $('detail-card');
const sels = id => tree.querySelectorAll('.sel[data-id="' + id + '"]');
const togFor = li => li.firstElementChild.firstElementChild;
const visibleRow = li => { for (let e = li; e && e !== tree; e = e.parentNode) { if (e.hidden) return false; } return true; };
let checks = 0; const ok = (c, m) => { assert.ok(c, m); checks++; };

ok(tree.querySelectorAll('li.line').length === 7, 'seven lines');
ok(tree.querySelectorAll('li.node').length === 69, '69 rows');
ok($('ov-grid').querySelectorAll('.ov-chip').length === 29, '29 chips');
ok(!card.hidden && $('d-title').textContent === data.nodes['H-0012'].title, 'hash selects H-0012');
ok($('d-purpose').textContent === data.nodes['H-0012'].purpose && $('d-result').textContent === data.nodes['H-0012'].result, 'sections filled');
ok(sels('H-0012').every(b => b.getAttribute('aria-current') === 'true'), 'aria-current on H-0012');
ok(visibleRow(tree.querySelector('li[data-id="H-0012"]')), 'H-0012 revealed');
ok($('d-rel').textContent.indexOf('源自：') === 0 && $('d-rel').querySelectorAll('button').length === 1, 'derived chip');

const h13 = tree.querySelector('li[data-id="H-0013"]'), t13 = togFor(h13);
ok(t13.getAttribute('aria-expanded') === 'false' && $(t13.getAttribute('aria-controls')).hidden, 'H-0013 collapsed by default');
click(t13);
ok(t13.getAttribute('aria-expanded') === 'true' && !$(t13.getAttribute('aria-controls')).hidden, 'toggle expands');
click(t13);
ok(t13.getAttribute('aria-expanded') === 'false', 'toggle collapses');

click($('expand-all'));
ok(tree.querySelectorAll('.tog, .line-tog').every(b => b.getAttribute('aria-expanded') === 'true'), 'expand all');
ok(tree.querySelectorAll('ul.kids').every(u => !u.hidden), 'all groups shown');
click($('collapse-all'));
ok(tree.querySelectorAll('.tog, .line-tog').every(b => b.getAttribute('aria-expanded') === 'false'), 'collapse all');

const e21 = tree.querySelector('li[data-id="H-0014"]').querySelector('.sel[data-id="E-0021"]');
click(e21);
ok($('d-title').textContent === data.nodes['E-0021'].title && $('d-did').textContent === data.nodes['E-0021'].did, 'E-0021 selected');
ok(sels('E-0021').length === 4 && sels('E-0021').every(b => b.getAttribute('aria-current') === 'true'), 'all four E-0021 rows current');
ok(sels('H-0012').every(b => !b.hasAttribute('aria-current')), 'previous current cleared');
ok($('d-rel').querySelectorAll('button').length === 4, 'tests chips');
ok(lastUrl === '#E-0021', 'hash updated');

click($('d-rel').querySelector('button[data-rel="H-0016"]'));
const h16 = tree.querySelector('li[data-id="H-0016"]');
ok($('d-title').textContent === data.nodes['H-0016'].title && visibleRow(h16) && h16.scrolled >= 1, 'relation chip selects and reveals');

click($('ov-grid').querySelector('.ov-chip[data-id="H-0028"]'));
ok($('d-title').textContent === data.nodes['H-0028'].title && visibleRow(tree.querySelector('li[data-id="H-0028"]')), 'overview chip');
ok($('ov-grid').querySelector('.ov-chip[data-id="H-0028"]').getAttribute('aria-current') === 'true', 'chip current');

const before = tree.querySelectorAll('.tog, .line-tog').map(b => b.getAttribute('aria-expanded')).join();
const f = $('filter');
f.value = 'LoRA'; input(f);
const shown = tree.querySelectorAll('li.node').filter(visibleRow).map(li => li.getAttribute('data-id'));
const expect = Object.keys(data.nodes).filter(id => (id + ' ' + data.nodes[id].title).toLowerCase().indexOf('lora') >= 0);
ok(expect.length > 0 && expect.every(id => shown.indexOf(id) >= 0), 'filter shows every match');
ok(shown.every(id => {
  const li = tree.querySelectorAll('li.node').find(x => x.getAttribute('data-id') === id && visibleRow(x));
  return expect.indexOf(id) >= 0 || li.querySelectorAll('li.node').some(d => expect.indexOf(d.getAttribute('data-id')) >= 0);
}), 'filter shows only matches and their ancestors');
ok($('no-match').hidden, 'no-match hidden');
f.value = 'zzzz'; input(f);
ok(tree.querySelectorAll('li.line').every(l => l.hidden) && !$('no-match').hidden, 'no match message');
f.value = ''; input(f);
ok(tree.querySelectorAll('li').every(l => !l.hidden) && $('no-match').hidden, 'filter cleared');
ok(tree.querySelectorAll('.tog, .line-tog').map(b => b.getAttribute('aria-expanded')).join() === before, 'expansion restored');

const th = $('theme');
ok(th.textContent === '配色：自动', 'theme label');
click(th); ok(html.getAttribute('data-theme') === 'light' && store['ht-theme'] === 'light', 'theme light');
click(th); ok(html.getAttribute('data-theme') === 'dark', 'theme dark');
click(th); ok(html.getAttribute('data-theme') === null && !('ht-theme' in store), 'theme auto');

narrow = true;
click(tree.querySelector('.sel[data-id="H-0001"]'));
ok(card.scrolled >= 1 && doc.activeElement === $('d-title'), 'narrow: scroll to details and focus heading');
click($('back-to-tree'));
ok(doc.activeElement === tree.querySelector('.sel[data-id="H-0001"]'), 'back to tree focuses row');
ok($('sr-status').textContent === '已显示 H-0001 的详情', 'live status');

Object.keys(data.nodes).forEach(id => {
  click(tree.querySelector('.sel[data-id="' + id + '"]'));
  assert.strictEqual($('d-title').textContent, data.nodes[id].title);
  assert.strictEqual($('d-result').textContent, data.nodes[id].result);
  assert.ok($('d-badges').querySelectorAll('.b').length === 2, id + ' badges');
  assert.ok($('d-id').textContent.indexOf(id + ' · ') === 0 && !/undefined/.test($('d-id').textContent), id + ' header');
  checks++;
});
console.log('DOM smoke test passed:', checks, 'checks');
