"""Verify the allocated MIG slice, after the PBS allocation check has passed."""
import json
import os
from pathlib import Path
import re
import subprocess


def selected_profile(listing, visible):
    devices = re.findall(r'MIG\s+(\S+)\s+Device[^\n]*UUID:\s*([^\)]+)\)', listing)
    if not devices:
        raise RuntimeError('nvidia-smi did not identify a MIG device')
    identifiers = [part.strip() for part in visible.split(',') if part.strip()]
    if len(identifiers) != 1:
        raise RuntimeError('exactly one CUDA-visible MIG device is required')
    if identifiers[0].startswith('MIG-'):
        matches = [item for item in devices if item[1].strip() == identifiers[0]]
        if len(matches) != 1:
            raise RuntimeError('CUDA-visible MIG UUID absent or ambiguous in nvidia-smi')
        return matches[0]
    if len(devices) == 1:
        return devices[0]
    raise RuntimeError('cannot map CUDA device index to a unique MIG profile')


def verify_gpu_profile(output):
    import torch
    output = Path(output)
    listing = subprocess.check_output(['nvidia-smi', '-L'], text=True)
    (output/'nvidia-smi.txt').write_text(listing)
    report = {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
              'torch_device_count': torch.cuda.device_count(), 'passed': False}
    try:
        profile, uuid = selected_profile(listing, report['cuda_visible_devices'])
        report.update(nvidia_smi_profile=profile, nvidia_smi_uuid=uuid,
                      torch_device_name=torch.cuda.get_device_name(0),
                      multiprocessor_count=torch.cuda.get_device_properties(0).multi_processor_count)
        if report['torch_device_count'] != 1 or profile != '2g.24gb' or not report['torch_device_name'].endswith('MIG 2g.24gb'):
            raise RuntimeError('requires actual MIG 2g.24gb for timing and bitwise comparisons')
        report['passed'] = True
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        (output/'hardware.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
