import json
import os
import re
import sys
from html.parser import HTMLParser

PAGE = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
VIS = os.path.join(HERE, "visible-text.txt")
VOID = {"meta", "link", "br", "hr", "img", "input", "source", "area", "base", "col", "embed", "param", "track", "wbr", "path"}
CJK = re.compile(r"[一-鿿]")

html = open(PAGE, encoding="utf-8").read()


class P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors, self.text, self.attrs_text = [], [], [], []
        self.scripts, self.cur_script, self.in_style = [], None, False
        self.ids = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        for k in ("placeholder", "aria-label", "title"):
            if a.get(k) and CJK.search(a[k]):
                self.attrs_text.append(a[k])
        if tag in VOID:
            return
        self.stack.append(tag)
        if tag == "script":
            self.cur_script = {"type": a.get("type"), "id": a.get("id"), "text": ""}
        if tag == "style":
            self.in_style = True

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.stack and self.stack[-1] == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> at {self.getpos()} stack={self.stack[-4:]}")
            if tag in self.stack:
                while self.stack and self.stack[-1] != tag:
                    self.stack.pop()
                self.stack.pop()
            return
        self.stack.pop()
        if tag == "script":
            self.scripts.append(self.cur_script)
            self.cur_script = None
        if tag == "style":
            self.in_style = False

    def handle_data(self, data):
        if self.cur_script is not None:
            self.cur_script["text"] += data
        elif not self.in_style and data.strip():
            self.text.append(data.strip())


p = P()
p.feed(html)
p.close()
assert not p.errors, p.errors
assert not p.stack, p.stack
assert len(p.ids) == len(set(p.ids)), "duplicate static ids"

data_scripts = [s for s in p.scripts if s["id"] == "tree-data"]
assert len(data_scripts) == 1
data = json.loads(data_scripts[0]["text"])
main_js = [s for s in p.scripts if s["type"] is None and "htInit" in s["text"]]
assert len(main_js) == 1
open(os.path.join(HERE, "page-main.js"), "w", encoding="utf-8").write(main_js[0]["text"])
json.dump(data, open(os.path.join(HERE, "page-data-roundtrip.json"), "w", encoding="utf-8"), ensure_ascii=False)

nodes = data["nodes"]
want = {f"H-{i:04d}" for i in range(1, 30)} | {f"E-{i:04d}" for i in range(1, 31)}
assert set(nodes) == want, set(nodes) ^ want
for nid, n in nodes.items():
    for k in ("title", "purpose", "did", "result"):
        assert isinstance(n.get(k), str) and n[k].strip(), (nid, k)

in_tree, rows, per_line = set(), 0, []
first_text = {}
for line in data["lines"]:
    seen = set()

    def walk(lst, parent):
        global rows
        for nid, kids in lst:
            assert nid in nodes, nid
            if nid.startswith("E-"):
                assert not kids, nid
                assert parent in nodes[nid]["tests"], (nid, parent)
            elif parent and parent.startswith("H-"):
                assert parent in nodes[nid]["derived"], (nid, parent)
            seen.add(nid)
            in_tree.add(nid)
            rows += 1
            walk(kids, nid)

    walk(line["tree"], None)
    h = sorted(x for x in seen if x.startswith("H"))
    e = sorted(x for x in seen if x.startswith("E"))
    per_line.append((line["name"], len(h), len(e)))
assert in_tree == want, want - in_tree
for nid, n in nodes.items():
    if nid.startswith("E-"):
        occurrences = []
        for line in data["lines"]:
            def w2(lst, parent):
                for x, kids in lst:
                    if x == nid:
                        occurrences.append(parent)
                    w2(kids, x)
            w2(line["tree"], None)
        assert sorted(occurrences) == sorted(n["tests"]), (nid, occurrences, n["tests"])

dash = [m.start() for m in re.finditer("[—–]", html)]
assert not dash, f"dashes at {dash[:5]}"

js_lits = []
for m in re.finditer(r"'((?:[^'\\]|\\.)*)'", main_js[0]["text"]):
    if CJK.search(m.group(1)):
        js_lits.append(m.group(1))

out = []
out += p.text
out += p.attrs_text
for line in data["lines"]:
    out += [line["name"], line["desc"]]
for nid in sorted(nodes):
    n = nodes[nid]
    out += [n["title"], n["purpose"], n["did"], n["result"]]
out += js_lits
with open(VIS, "w", encoding="utf-8") as f:
    f.write("\n".join(s.replace("\n", " ") for s in out) + "\n")

print("html parse: ok, balanced, static ids unique:", len(p.ids))
print("json: ok, nodes:", len(nodes), "H:", sum(1 for k in nodes if k[0] == "H"), "E:", sum(1 for k in nodes if k[0] == "E"))
print("tree rows (with duplicates):", rows)
for name, h, e in per_line:
    print(f"  {name}: H {h}, E {e}, total {h + e}")
print("dashes U+2014/U+2013 in page: 0")
print("visible text lines:", len(out), "->", VIS)
print("size bytes:", os.path.getsize(PAGE))
