import json

import pytest

from v5_train import ARMS,CONTRACT,GRID,write_json
from v5_calibration_report import summarize


def fixtures(tmp_path):
    paths = []
    for arm in ARMS:
        for lr,nll in zip(GRID,(1.009,1.005,1.0)):
            path = tmp_path/f'{arm}-{lr}.json'
            write_json(path,{'status':'complete','validator':{'passed':True},
                            'phase':'calibration','contract':CONTRACT,'seed':0,'arm':arm,
                            'metrics':{'learning_rate':lr,'calibration.nll':nll,
                            'plan.order_sha256':'paired-order','plan.steps':1400,
                            'plan.input_tokens':200041,'train.steps':1400,
                            'train.input_tokens':200041,'elapsed_s':650.}})
            paths.append(path)
    return paths


def test_initial_grid_requires_raw_minimum_extension_before_tie_lock(tmp_path):
    paths = fixtures(tmp_path)
    report = summarize(paths)
    assert report['required_extensions'] == dict.fromkeys(ARMS,3e-3)
    assert report['provisional_initial_choices'] == dict.fromkeys(ARMS,1e-4)
    assert report['paired_plan']['input_tokens'] == 200041
    assert len(report['rows']) == 9


def test_grid_report_rejects_incomplete_plan_and_different_pairing(tmp_path):
    paths = fixtures(tmp_path)
    value = json.loads(paths[0].read_text())
    value['metrics']['train.steps'] = 100
    write_json(paths[0],value)
    with pytest.raises(ValueError,match='did not complete'):
        summarize(paths)
    value['metrics']['train.steps'] = 1400
    value['metrics']['plan.order_sha256'] = 'different'
    write_json(paths[0],value)
    with pytest.raises(ValueError,match='different sample'):
        summarize(paths)
