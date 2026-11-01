import json
from types import SimpleNamespace

import pytest

import mig_profile_guard as guard


GPU = 'GPU 0: NVIDIA GH200 120GB (UUID: GPU-physical)\n'
GOOD = GPU+'  MIG 2g.24gb     Device  0: (UUID: MIG-assigned)\n'
BAD = GPU+'  MIG 1g.24gb     Device  0: (UUID: MIG-assigned)\n'


def test_resolves_allocated_uuid_and_site_spacing():
    assert guard.allocated_profile(GOOD,'MIG-assigned')['profile'] == '2g.24gb'


@pytest.mark.parametrize('visible',[None,'0','MIG-first,MIG-second'])
def test_rejects_unknown_or_multiple_allocated_devices(visible):
    with pytest.raises(ValueError,match='one scheduler-provided MIG UUID'):
        guard.allocated_profile(GOOD,visible)


def test_rejects_mismatched_or_duplicate_uuid():
    with pytest.raises(ValueError,match='missing or ambiguous'):
        guard.allocated_profile(GOOD,'MIG-other')
    with pytest.raises(ValueError,match='missing or ambiguous'):
        guard.allocated_profile(GOOD+GOOD,'MIG-assigned')


@pytest.mark.parametrize('listing,code,profile',[(GOOD,0,'2g.24gb'),
    (BAD+'  MIG 2g.24gb Device 1: (UUID: MIG-unrelated)\n',3,'1g.24gb')])
def test_cli_gates_allocated_slice_and_preserves_failure_evidence(
        monkeypatch,tmp_path,listing,code,profile):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','MIG-assigned')
    monkeypatch.setenv('PBS_JOBID','123.opbs')
    monkeypatch.setattr(guard.subprocess,'run',lambda *a,**k:
                        SimpleNamespace(stdout=listing,stderr='',returncode=0))
    assert guard.main(['--output',str(tmp_path)]) == code
    record = json.loads((tmp_path/'mig-profile.json').read_text())
    assert record['profile'] == profile and record['passed'] is (code==0)
    assert record['uuid'] == 'MIG-assigned'
    assert (tmp_path/'nvidia-smi-L.txt').read_text() == listing
    assert guard.os.environ['CUDA_VISIBLE_DEVICES'] == 'MIG-assigned'
