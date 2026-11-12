"""Missing endpoint-mechanics checks only. No cloud/test/training imports or calls.

Writes exclusively into a NEW results folder. Policies receive no audit labels.
Existing stagewise/saturation/swap/exposure tables are hash-checked and reused.
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
POST = HERE.parents[1] / 'experiments/posthoc100'
SOURCE = {}
CHECKS = []


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def used(path, expected=None):
    path = Path(path)
    actual = sha(path)
    require(expected is None or actual == expected, 'Source hash mismatch: ' + str(path))
    SOURCE[str(path)] = {'sha256': actual, 'bytes': path.stat().st_size}
    return path


def read(path, expected=None):
    return json.loads(used(path, expected).read_text(encoding='utf-8'))


def arrays(path, expected):
    with np.load(used(path, expected), allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def save(out, name, obj):
    with (out / name).open('x', encoding='utf-8') as f:
        json.dump(plain(obj), f, indent=2, allow_nan=False)


def table(out, name, rows):
    require(bool(rows), 'Empty table: ' + name)
    with (out / name).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(plain(rows))


def global_selection(score, k):
    require(score.ndim == 1 and np.isfinite(score).all(), 'Invalid ranking scores')
    order = np.lexsort((np.arange(len(score)), -score.astype(np.float64)))
    keep = np.zeros(len(score), bool)
    keep[order[:k]] = True
    rank = np.empty(len(score), np.int64)
    rank[order] = np.arange(1, len(score) + 1)
    return keep, rank, float(score[order[k-1]])


def class_selection(score, observed, k, classes):
    # No reference labels, hardness masks, validation or test information here.
    n = len(score)
    counts = np.bincount(observed, minlength=classes).astype(np.int64)
    products = k * counts
    quota = products // n
    priority = np.lexsort((np.arange(classes), -(products % n)))
    quota[priority[:k-int(quota.sum())]] += 1
    keep = np.zeros(n, bool)
    for c in range(classes):
        ids = np.flatnonzero(observed == c)
        order = np.lexsort((ids, -score[ids].astype(np.float64)))
        keep[ids[order[:quota[c]]]] = True
    require(int(keep.sum()) == k and np.all(quota <= counts), 'Quota conservation')
    return keep, counts, quota


def check(name, condition):
    require(condition, name)
    CHECKS.append({'check': name, 'status': 'PASS'})


def synthetic_tests():
    keep, rank, cutoff = global_selection(np.ones(10), 7)
    check('Global deterministic ties retain first local IDs', np.array_equal(np.flatnonzero(keep), np.arange(7)))
    check('Rank inverse and cutoff', np.array_equal(rank, np.arange(1, 11)) and cutoff == 1)
    obs = np.array([0]*5+[1]*3+[2]*2)
    keep, count, quota = class_selection(np.ones(10), obs, 7, 4)
    check('Largest remainders and empty class', np.array_equal(quota, [4, 2, 1, 0]))
    check('Within-class local-ID ties', np.array_equal(np.flatnonzero(keep), [0, 1, 2, 3, 5, 6, 8]))
    keep, _, quota = class_selection(np.ones(4), np.array([0, 0, 1, 1]), 1, 2)
    check('Remainder ties use ascending observed class', np.array_equal(quota, [1, 0]))
    keep, _, _ = class_selection(np.array([.1, .8, .3, .5]), np.array([0, 0, 1, 1]), 2, 2)
    check('Per-class descending score', np.array_equal(np.flatnonzero(keep), [1, 3]))
    check('Integer floor coverage equals frozen float implementation', all((7*n)//10 == int(np.floor(.7*n)) for n in [10, 18388, 17614]))


def main():
    PREP = Path(os.environ['SSTRE_AUDIT_EVIDENCE'])
    SCI = PREP / 'final_scientific_analysis_20261027_v1'
    EXPOSURE = PREP / 'audits/main100_comparison_exposure_20261027_02'
    PAIR = PREP / 'audits/posthoc_seed43_pair_20261028_01'
    STAGE43 = Path(os.environ['SSTRE_STAGE43_EVIDENCE'])
    policy = read(HERE/'EXPLORATORY_POLICY_PREDECLARATION.json')
    require(policy['coverage'] == .7, 'Coverage not locked')
    synthetic_tests()
    # Reuse these accepted tables instead of re-running their analyses.
    package = read(SCI/'PACKAGE_SHA256.json')
    tables = {}
    for name in ['frozen_endpoint_reliability_seven.csv', 'endpoint_score_saturation_and_cutoffs_AUDIT_ONLY.csv',
                 'same_evidence_coefficient_allocation_AUDIT_ONLY.csv', 'signal_rank_correlations_AUDIT_ONLY.csv',
                 'stagewise_saved_signal_diagnostics_AUDIT_ONLY.csv', 'hard_tail_cumulative_exposure_existing.csv']:
        relative = 'tables/' + name
        with used(SCI/relative, package[relative]['sha256']).open(encoding='utf-8-sig', newline='') as f:
            tables[name] = list(csv.DictReader(f))

    split = arrays(PREP / 'dataset_split_v2.npz',
                   '2cef9dc2d707c029a70276fcc6736401a282f9755abc4cb8ed00f581269f33f4')
    mask10 = arrays(PREP/'audits/phase6_seed42_audit_20261024_01/results_02/frozen_audit_masks_AUDIT_ONLY.npz',
                    'fc935bf197ad4810e237279b396c3ac2bc7bef6b5ded129fc33938b1224509f0')
    mask100 = arrays(PREP/'audits/main100_reliability_20261027_01/results_02/CIFAR100_hard_clean_tail_MASK_AUDIT_ONLY.npz',
                     'e21bfa307b87af4cd51cc270a30d27967d05adc197bcebbf2114567ad8447e47')
    transfer = read(PREP/'dataset_transfers/cifar100n_v1_20261026_01/TRANSFER_REPORT.json')
    def hundred(name):
        entry = next(r for r in transfer['files'] if r['remote_path'].endswith('/'+name))
        return arrays(entry['local_readback'], entry['sha256'])
    train100, audit100 = hundred('train.npz'), hundred('AUDIT_ONLY.npz')
    views = {}
    for dataset, ids, obs, truth, mask in [
        ('cifar10n', split['lt_indices'], split['observed_labels'], split['clean_labels_AUDIT_ONLY'], mask10),
        ('cifar100n', train100['original_ids'], train100['observed_labels'], audit100['clean_labels'], mask100)]:
        cfg = read(POST/'reviewed_protocol/datasets'/f'{dataset}.json')
        check(dataset+' mask original ID alignment', np.array_equal(ids, mask['original_ids']))
        hard = mask['hard_clean_tail_EXPLORATORY_AUDIT_ONLY']
        clean = obs == truth
        groups = {g: np.isin(truth, cs) for g, cs in cfg['evaluation_groups'].items()}
        check(dataset+' hard mask clean-tail identity', bool((clean & groups['tail'])[hard].all()))
        views[dataset] = dict(ids=ids, observed=obs, truth=truth, clean=clean, hard=hard,
                              groups=groups, classes=cfg['num_classes'])

    runs = []
    for dataset in views:
        for method in ['fixed', 'adaptive']:
            directory = EXPOSURE/'evidence'/f'{dataset}_{method}'
            complete = read(directory/'COMPLETE.json')
            require(complete['status'] == 'PASS_WEIGHT_EVIDENCE', 'Unaccepted evidence')
            record = next(x for x in complete['archives'] if x['epoch'] == 100)
            a = arrays(directory/'epoch_100.npz', record['derived']['sha256'])
            expected = next(r for r in tables['frozen_endpoint_reliability_seven.csv'] if r['dataset']==dataset and
                            r['method']==('Fixed' if method=='fixed' else 'StageAware'))
            runs.append((dataset, 42, method, complete['run_id'], a, expected))
    for method, directory in [('fixed', PAIR/'fixed'), ('adaptive', STAGE43)]:
        receipt = read(directory/'AUDIT_FETCH_RECEIPT.json')
        require(receipt['status'] == 'PASS_HASH_VERIFIED_INPUTS', 'Unverified seed43 evidence')
        a = arrays(directory/'audit_inputs/trajectory_epoch_100.npz', receipt['files']['trajectory_epoch_100.npz']['sha256'])
        audit = read(directory/'FROZEN_RELIABILITY_AUDIT.json')
        require(audit['status'] == 'PASS_FROZEN_RELIABILITY_AUDIT', 'No accepted seed43 audit')
        runs.append(('cifar10n', 43, method, receipt['source_directory'].split('/')[-1], a, audit['primary']))

    out = Path(os.environ['SSTRE_AUDIT_OUTPUT'])
    out.mkdir(exist_ok=False)
    policy_rows, sample_rows, quota_rows, overlap_rows, rank_rows = [], [], [], [], []
    selections = {}
    for dataset, seed, method, run_id, a, expected in runs:
        v = views[dataset]; n = len(v['ids']); k = (7*n)//10
        check(run_id+' frozen IDs/observed labels', np.array_equal(a['original_ids'], v['ids']) and np.array_equal(a['observed_labels'], v['observed']))
        check(run_id+' causal endpoint', a['C_epoch'].item()==99 and np.array_equal(a['T_epochs'], np.arange(94, 99)) and a['N_epoch'].item()==95)
        coef = np.array([23/70, 24/70, 23/70]) if method=='fixed' else np.array([4/15, 7/15, 4/15])
        R = np.clip(sum(c*a[key].astype(np.float64) for c,key in zip(coef, ['C','T','N_soft'])), 0, 1).astype(np.float32)
        check(run_id+' unchanged exact fusion', np.array_equal(coef, a['coefficients']) and np.array_equal(R, a['reliability']))
        keep, ranks, cutoff = global_selection(R, k)
        alternative, counts, quota = class_selection(R, v['observed'], k, v['classes'])
        # Policy is complete before either mask is evaluated using audit references.
        hard, clean = v['hard'], v['clean']
        frr_key = 'hard_clean_tail_false_rejection_EXPLORATORY' if seed==42 else 'hard_clean_tail_false_rejection'
        check(run_id+' reproduces frozen rejection and count', int(keep.sum())==int(expected['selected_count']) and
              abs(float((~keep[hard]).mean())-float(expected[frr_key]))<1e-14)
        check(run_id+' reproduces frozen clean precision', abs(float(clean[keep].mean())-float(expected['selection_precision_clean']))<1e-14)
        check(run_id+' exact total selected counterfactual', int(alternative.sum())==k)
        selections[(dataset,seed,method)] = dict(keep=keep, alternative=alternative, ranks=ranks, R=R, cutoff=cutoff)
        for c in range(v['classes']):
            cmask = v['observed']==c
            quota_rows.append(dict(dataset=dataset, seed=seed, method=method, observed_class=c,
                samples=int(counts[c]), global_selected=int(keep[cmask].sum()), class_quota=int(quota[c]),
                global_class_coverage=float(keep[cmask].mean()) if counts[c] else None))
        for name, selection in [('frozen_global70',keep),('exploratory_proportional_observed_class70',alternative)]:
            row = dict(dataset=dataset,seed=seed,method=method,run_id=run_id,policy=name,selected=k,
                hard_count=int(hard.sum()),hard_rejected=int((hard & ~selection).sum()),
                hard_rejection_rate=float((~selection[hard]).mean()),
                noisy_selected=int((~clean & selection).sum()),noisy_total=int((~clean).sum()),
                noisy_retention_rate=float(selection[~clean].mean()),
                clean_precision=float(clean[selection].mean()),clean_recall=float(selection[clean].mean()),
                selection_swaps_from_global=int((selection & ~keep).sum()))
            for group, g in v['groups'].items():
                row['clean_'+group+'_rejection_rate'] = float((~selection[clean & g]).mean())
            policy_rows.append(row)
        h_reject = hard & ~keep
        rank_rows.append(dict(dataset=dataset,seed=seed,method=method,cutoff=cutoff,
            hard_rejected=int(h_reject.sum()),hard_mean_R=float(R[hard].mean(dtype=np.float64)),
            rejected_rank_min=int(ranks[h_reject].min()),rejected_rank_median=float(np.median(ranks[h_reject])),
            rejected_rank_max=int(ranks[h_reject].max()),
            rejected_score_margin_median=float(np.median(R[h_reject].astype(np.float64)-cutoff)),
            rejected_score_margin_min=float(np.min(R[h_reject].astype(np.float64)-cutoff)),
            rejected_score_margin_max=float(np.max(R[h_reject].astype(np.float64)-cutoff)),
            hard_rejected_T1=int((h_reject & (a['T']==1)).sum()),
            hard_at_cutoff=int((hard & (R==cutoff)).sum()),all_at_cutoff=int((R==cutoff).sum())))
        for i in np.flatnonzero(hard):
            sample_rows.append(dict(dataset=dataset,seed=seed,method=method,local_id=int(i),original_id=int(v['ids'][i]),
                observed_class=int(v['observed'][i]),reference_class_AUDIT_ONLY=int(v['truth'][i]),
                C=float(a['C'][i]),T=float(a['T'][i]),N_soft=float(a['N_soft'][i]),R=float(R[i]),
                global_rank=int(ranks[i]),selected_total=k,rank_minus_K=int(ranks[i]-k),
                global_cutoff=cutoff,score_minus_cutoff=float(R[i])-cutoff,
                rejected_global=not bool(keep[i]),rejected_class_counterfactual=not bool(alternative[i])))

    for dataset, seed in [('cifar10n',42),('cifar100n',42),('cifar10n',43)]:
        f,s = (selections[(dataset,seed,m)] for m in ['fixed','adaptive'])
        hard = views[dataset]['hard']; rf = hard & ~f['keep']; rs = hard & ~s['keep']
        overlap_rows.append(dict(dataset=dataset,seed=seed,hard_count=int(hard.sum()),
            rejected_both=int((rf & rs).sum()),rejected_fixed_only=int((rf & ~rs).sum()),
            rejected_stage_only=int((rs & ~rf).sum()),selected_both=int((hard & ~rf & ~rs).sum()),
            rejected_union=int((rf | rs).sum()),rejection_jaccard=float((rf & rs).sum()/(rf | rs).sum()),
            all_sample_selected_both=int((f['keep'] & s['keep']).sum()),
            fixed_rejected_original_ids=views[dataset]['ids'][rf].tolist(),
            stage_rejected_original_ids=views[dataset]['ids'][rs].tolist()))

    # Reuse, do not recalculate, previous signal/trajectory diagnostics.
    reuse = {
        'endpoint_saturation': tables['endpoint_score_saturation_and_cutoffs_AUDIT_ONLY.csv'],
        'endpoint_swaps': [r for r in tables['same_evidence_coefficient_allocation_AUDIT_ONLY.csv'] if int(r['loss_epoch'])==100],
        'all_stage_swaps': tables['same_evidence_coefficient_allocation_AUDIT_ONLY.csv'],
        'endpoint_correlations': [r for r in tables['signal_rank_correlations_AUDIT_ONLY.csv'] if int(r['loss_epoch'])==100],
        'temporal_stage_diagnostics': [r for r in tables['stagewise_saved_signal_diagnostics_AUDIT_ONLY.csv'] if r['score']=='T' and r['group_AUDIT_ONLY']=='all'],
        'exposure': tables['hard_tail_cumulative_exposure_existing.csv'],
        'seed43_exposure': 'Unavailable over full100 epochs in accepted local paired-audit evidence; not estimated or substituted.'
    }
    table(out,'SELECTION_COUNTERFACTUAL_AUDIT_ONLY.csv',policy_rows)
    table(out,'OBSERVED_CLASS_QUOTAS.csv',quota_rows)
    table(out,'HARD_SAMPLE_RANK_MARGINS_AUDIT_ONLY.csv',sample_rows)
    table(out,'RANK_MARGIN_SUMMARY_AUDIT_ONLY.csv',rank_rows)
    table(out,'REJECTION_OVERLAP_AUDIT_ONLY.csv',[{k:v for k,v in r.items() if not k.endswith('_ids')} for r in overlap_rows])
    save(out,'REJECTED_ORIGINAL_IDS_AUDIT_ONLY.json',overlap_rows)
    save(out,'REUSED_DIAGNOSTICS.json',reuse)
    for path, record in SOURCE.items():
        require(sha(path)==record['sha256'], 'Input changed during audit: '+path)
    save(out,'INPUT_SHA256.json', SOURCE)
    save(out,'LOCAL_CHECKS.json',dict(passed=len(CHECKS), failed=0, skipped=0, checks=CHECKS))
    result = dict(status='PASS_LOCAL_ENDPOINT_MECHANICS',source_files=len(SOURCE),
        cloud_connections=0,cloud_submissions=0,remote_writes=0,official_test_access=False,
        completed_analyses_reexecuted=False,original_results_modified=False,
        policy_comparison=policy_rows,overlap=overlap_rows,rank_margins=rank_rows)
    save(out,'FINDINGS.json',result)
    save(out,'OUTPUT_SHA256.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(dict(status=result['status'],checks=len(CHECKS),policy_comparison=policy_rows,
        overlap=[{k:v for k,v in r.items() if not k.endswith('_ids')} for r in overlap_rows],rank_margins=rank_rows),indent=2))


if __name__ == '__main__':
    main()
