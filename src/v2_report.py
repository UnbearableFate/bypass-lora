"""Project measured E=16 smoke throughput to planning costs, not efficacy."""
import argparse
import json
from pathlib import Path


def costs(results):
    arms = {}
    for arm, result in results.items():
        train = result['train']
        throughput = train['tokens_per_second']
        if throughput <= 0:
            raise ValueError('nonpositive throughput')
        arms[arm] = {'step_seconds': train['stable_step_seconds'],
                     'tokens_per_second': throughput,
                     'peak_allocated_bytes': train['peak_allocated_bytes'],
                     'train_4M_node_hours': 4_000_000/throughput/3600*.25,
                     'train_0_2M_node_hours': 200_000/throughput/3600*.25,
                     'six_control_eval_seconds': result.get('six_control_eval_seconds'),
                     'checkpoint_save_reload_seconds': result.get('checkpoint_save_reload_seconds')}
    ranges = {}
    for backend in ('kernel_matrix', 'soft_slot'):
        values = [a['train_0_2M_node_hours'] for name, a in arms.items() if backend in name]
        if len(values) != 4:
            raise ValueError('need all four feature/fusion choices per backend')
        ranges[backend] = {'two_LRs_min_node_hours': 2*min(values), 'two_LRs_max_node_hours': 2*max(values)}
    return {'arms': arms, 'V2_8x4M_train_node_hours': sum(a['train_4M_node_hours'] for a in arms.values()),
            'calibration_2backends_x2LRs_x0_2M': ranges,
            'calibration_total_min_node_hours': sum(v['two_LRs_min_node_hours'] for v in ranges.values()),
            'calibration_total_max_node_hours': sum(v['two_LRs_max_node_hours'] for v in ranges.values()),
            'limitations': 'Extrapolation from small fixed batches; training-only cost excludes startup, data variation, evaluation frequency, checkpoint schedule and PBS queue time. Input tokens count support plus question tokens actually forwarded; one-token answer is a CE target, not a forwarded input. Calibration feature/fusion choice is not fixed here; reported bounds cover all four choices per family. No confidence interval or authorization for V2 training.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('output')
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(costs(json.loads(Path(args.results).read_text())), indent=2)+'\n')


if __name__ == '__main__':
    main()
