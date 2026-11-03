"""Require the allocated MIG UUID to match its comparison's requested profile.

Run after the site's allocation check and before model imports. This guard
preserves CUDA_VISIBLE_DEVICES and saves nvidia-smi evidence even on rejection.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess


def allocated_profile(listing, visible):
    if not visible or not visible.startswith('MIG-') or ',' in visible:
        raise ValueError('one scheduler-provided MIG UUID is required in CUDA_VISIBLE_DEVICES')
    entries = re.finditer(r'^\s*MIG\s+(\S+)\s+Device\s+\d+:\s+\(UUID:\s*(MIG-[^)\s]+)\)\s*$',
                          listing, flags=re.MULTILINE)
    matches = [match for match in entries if match.group(2) == visible]
    if len(matches) != 1:
        raise ValueError('allocated MIG UUID is missing or ambiguous in nvidia-smi -L')
    match = matches[0]
    return {'profile':match.group(1),'uuid':match.group(2),'selected_line':match.group(0).strip()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-profile',choices=('1g.24gb','2g.24gb'),default='2g.24gb')
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    record = {'expected_profile':args.expected_profile,'passed':False,
              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'pbs_job_id':os.environ.get('PBS_JOBID'),'host':socket.gethostname()}
    try:
        result = subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=10)
        (out/'nvidia-smi-L.txt').write_text(result.stdout)
        record['nvidia_smi_returncode'] = result.returncode
        if result.returncode != 0:
            raise ValueError('nvidia-smi -L failed: '+result.stderr.strip())
        record.update(allocated_profile(result.stdout,record['cuda_visible_devices']))
        if record['profile'] != record['expected_profile']:
            raise ValueError('allocated MIG '+record['profile']+';required MIG '+record['expected_profile'])
        record['passed'] = True
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        record['error'] = str(exc)
    (out/'mig-profile.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    return 0 if record['passed'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
