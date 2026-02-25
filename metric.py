import os
import wandb
import pandas as pd

project = "carecall_memory"
ids = ["run_20260222_3502e2be", "run_20260222_8d8604ad","run_20260223_0be065b3"]
output_exp_name = "qwen3_14b_oss_eval_dynamic_huatuo_meddialog_generalibility"

keys = [
    'agent/score/all_source_train_reward',
    'actor/entropy',
    'agent/think_len/step_think_len_all_source',
    'val/test_score/memory/cmtmedqa_avgpbo0',
    'val/test_score/memory/huatuo26m_avgpbo0',
    'val/test_score/memory/meddialog_cn_avgpbo0',
    'val/test_score/memory/medical_dialogue_avgpbo0',
]

api = wandb.TrackingApi()

output_dir = os.path.join("outputs", output_exp_name)
os.makedirs(output_dir, exist_ok=True)

dfs_per_run = []
for run_id in ids:
    run = api.run(project=project, run_id=run_id)
    hist = run.history(name=keys)
    steps = hist.pop('step', [])
    df = pd.DataFrame(hist, index=steps)
    df.index.name = 'step'
    dfs_per_run.append(df)

# re-index each df so steps are globally continuous across runs
offset = 0
for i, df in enumerate(dfs_per_run):
    new_index = range(offset, offset + len(df))
    dfs_per_run[i] = df.set_index(pd.Index(new_index, name='step'))
    offset += len(df)

merged = pd.concat(dfs_per_run, axis=0)

for key in keys:
    if key not in merged.columns:
        print(f"[WARN] key '{key}' not found in history, skipping")
        continue
    col_df = merged[[key]].dropna()
    safe_name = key.replace('/', '_')
    out_path = os.path.join(output_dir, f"{safe_name}.csv")
    col_df.to_csv(out_path)
    print(f"Saved {out_path}  ({len(col_df)} rows)")

print(f"\nAll done. Total steps: {len(merged)}, output dir: {output_dir}")
