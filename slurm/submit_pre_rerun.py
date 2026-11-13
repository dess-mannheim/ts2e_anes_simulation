"""Submits the complete pre-election re-run (simulations, annotation, evaluation) as Slurm dependency chains.

Run from the slurm/ directory:  python submit_pre_rerun.py [--dry-run]

Each "slot" is one GPU that runs one job at a time: a job waits (afterany) for the previous job of its slot(s).
With 2 slots on compute1 and 4 on compute2, at most 6 GPUs are used and one GPU on compute1 stays idle.
Jobs are placed greedily using rough runtime estimates: seed 42 of every model first, then the remaining seeds,
then one annotation job per simulated model (after all its seeds), then the evaluation notebook (CPU only).
"""
import subprocess
import sys

DRY_RUN = '--dry-run' in sys.argv

SLOTS = {'S1': 'compute1', 'S2': 'compute1', 'S3': 'compute2', 'S4': 'compute2', 'S5': 'compute2', 'S6': 'compute2'}

# model -> (GPUs, estimated hours per seed)
MODELS = {
    'meta-llama/Llama-3.3-70B-Instruct': (2, 1.6),
    'allenai/Olmo-3.1-32B-Instruct': (1, 1.3),
    'allenai/Olmo-3-7B-Instruct': (1, 0.7),
    'meta-llama/Llama-3.1-8B-Instruct': (1, 0.4),
    'Qwen/Qwen3-VL-8B-Instruct': (1, 0.35),
    'Qwen/Qwen3-VL-30B-A3B-Instruct': (1, 0.35),
}
FIRST_SEED = 42
OTHER_SEEDS = [43, 44, 45, 46]
ANNOTATION_HOURS = 2.0

slot_free = {s: 0.0 for s in SLOTS}  # estimated hours from now until the slot is free
slot_last_job = {s: None for s in SLOTS}


def sbatch(args, tag):
    cmd = ['sbatch', '--parsable',
           f'--output=slurm_logs/GABMpre_%j_{tag}.out', f'--error=slurm_logs/GABMpre_%j_{tag}.err'] + args
    if DRY_RUN:
        return f'<{tag}>'
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(';')[0]


def place(n_gpus, hours, script_args, tag, ready=0.0, extra_deps=()):
    """Put a job on the n_gpus slots of one node that free up earliest."""
    candidates = []
    for node in sorted(set(SLOTS.values())):
        node_slots = sorted((s for s in SLOTS if SLOTS[s] == node), key=lambda s: slot_free[s])[:n_gpus]
        start = max([ready] + [slot_free[s] for s in node_slots])
        candidates.append((start, node, node_slots))
    start, node, chosen = min(candidates)

    deps = sorted({slot_last_job[s] for s in chosen if slot_last_job[s]} | set(extra_deps))
    args = [f'--job-name=GABMpre_{tag.split("_")[0]}', f'--nodelist={node}', f'--gres=gpu:{n_gpus}']
    if deps:
        args.append(f'--dependency=afterany:{":".join(deps)}')
    job_id = sbatch(args + script_args, tag)

    for s in chosen:
        slot_free[s] = start + hours
        slot_last_job[s] = job_id
    print(f'{job_id:>8} {tag:34s} {node} {",".join(chosen):6s} start~+{start:4.1f}h end~+{start + hours:4.1f}h')
    return job_id, start + hours


sim_jobs = {model: [] for model in MODELS}  # model -> [(job id, estimated end)]

# 1) one seed for every model, 2) remaining seeds (longest jobs first)
for seeds in ([FIRST_SEED], OTHER_SEEDS):
    for model, (n_gpus, hours) in sorted(MODELS.items(), key=lambda kv: -kv[1][0] * kv[1][1]):
        for seed in seeds:
            short = model.split('/')[1]
            sim_jobs[model].append(place(
                n_gpus, hours, ['run_qstn_vote_prediction.sh', model, str(seed)], f'sim_{short}_seed{seed}'
            ))

# 3) annotation of each model's open responses once all of its seeds are done
annotation_ids = []
for model in sorted(MODELS, key=lambda m: max(end for _, end in sim_jobs[m])):
    short = model.split('/')[1]
    job_id, _ = place(
        1, ANNOTATION_HOURS, ['run_qstn_annotation.sh', short], f'annot_{short}',
        ready=max(end for _, end in sim_jobs[model]), extra_deps=[j for j, _ in sim_jobs[model]],
    )
    annotation_ids.append(job_id)

# 4) evaluation notebook (no GPU) after all annotations
eval_id = sbatch([f'--dependency=afterany:{":".join(annotation_ids)}', 'run_eval.sh'], 'eval')
print(f'{eval_id:>8} eval (CPU); estimated total ~{max(slot_free.values()) + 0.75:.1f}h')
