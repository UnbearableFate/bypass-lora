#!/bin/bash
set -euo pipefail
out="$1"
mkdir -p "$out"
/usr/bin/python3 -I /work/xg24i002/x10041/.codex/skills/miyabi-pbs-cmd/scripts/context.py --target-system Miyabi-G --require-compute > "$out/allocation.json"
qstat -f "$PBS_JOBID" > "$out/job.txt"
hostname > "$out/hostname.txt"
nvidia-smi > "$out/nvidia-smi.txt"
nvidia-smi -L > "$out/gpu-list.txt"
source /etc/profile.d/modules.sh
module avail cuda > "$out/cuda-modules.txt" 2>&1
if curl -I --connect-timeout 10 --max-time 20 https://download.pytorch.org/whl/ > "$out/network.txt" 2>&1; then
  network=true
else
  network=false
fi
cat > "$out/manifest.json" <<EOF
{"status":"complete","metrics":{"hardware_probe.pass":true,"compute_network.available":$network},"artifacts":[{"path":"$out/allocation.json","role":"allocation"},{"path":"$out/job.txt","role":"allocation_state"},{"path":"$out/nvidia-smi.txt","role":"hardware"},{"path":"$out/gpu-list.txt","role":"hardware"},{"path":"$out/cuda-modules.txt","role":"environment"},{"path":"$out/network.txt","role":"network"}],"validator":{"name":"probe_environment.sh","passed":true}}
EOF
