#!/bin/bash
# Rebuild docs/explanations/hypothesis-tree.html from the current records and nodes.py.
# Control work on the login node: dumps records with `rl --json show`, runs build.py and validate.py,
# then removes the regenerable byproducts. Edit nodes.py (prose, RUN_STATE) and the counts in build.py first.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
rl=/work/xg24i002/x10041/smooth-research/dist/claude/plugins/smooth-research/bin/rl
page="$root/docs/explanations/hypothesis-tree.html"
cd "$root"
mkdir -p "$here/rec"
read -r nh ne < <(python3 -I -c 'import re,sys; m=re.search(r"N_H, N_E = (\d+), (\d+)", open(sys.argv[1]).read()); print(m.group(1), m.group(2))' "$here/build.py")
for k in H E; do
  if [ "$k" = H ]; then n=$nh; else n=$ne; fi
  for ((i = 1; i <= n; i++)); do
    id=$(printf "%s-%04d" "$k" "$i")
    "$rl" --json show "$id" > "$here/rec/$id.json"
  done
done
dv=$("$rl" status | head -1 | grep -oE 'data_version [0-9]+' | awk '{print $2}')
(cd "$here" && python3 -I build.py "$page" "$dv" && python3 -I validate.py "$page")
/home/x10041/.claude/skills/write/scripts/check-punctuation.sh --lang zh "$here/visible-text.txt"
rm -rf "$here/rec" "$here/page-data.json" "$here/page-data-roundtrip.json" "$here/prose-dump.txt" "$here/visible-text.txt"
echo "rebuilt $page (data_version $dv)"
