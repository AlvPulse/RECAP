"""
Parse training_log*.txt and plot training curves.
Handles both UTF-8 and UTF-16 encoded log files.
Run: python plot_training_curve.py
"""

import re, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_FILES = ['training_log.txt', 'training_log2.txt', 'training_log3.txt', 'training_log5.txt']


def parse_log(path):
    """Parse SB3 training log, return list of dicts per iteration."""
    try:
        raw = open(path, 'rb').read()
        # Detect UTF-16 BOM
        if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
            text = raw.decode('utf-16')
        else:
            text = raw.decode('utf-8', errors='replace')
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return []

    records = []
    current = {}
    for line in text.splitlines():
        line = line.strip()
        # Match SB3 stats table: "| key | value |"
        m = re.match(r'\|\s+(\S[\w/\s]+?)\s+\|\s+([\d.\-e]+)\s+\|', line)
        if m:
            key = m.group(1).strip().replace(' ', '_').replace('/', '_')
            try:
                current[key] = float(m.group(2))
            except ValueError:
                pass
        # Save when we see a block separator that signals end of an iteration's table
        if line.startswith('---') and current.get('iterations') and 'explained_variance' in current:
            records.append(dict(current))
            current = {}

    # Flush last block
    if current.get('iterations') and 'explained_variance' in current:
        records.append(dict(current))

    return records


def main():
    all_records = {}
    for f in LOG_FILES:
        if os.path.exists(f):
            recs = parse_log(f)
            if recs:
                all_records[f] = recs
                print(f"{f}: {len(recs)} iterations parsed")

    if not all_records:
        print("No training logs found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('PPO Training Curves — Fixed Environment (All Bugs Resolved)', fontsize=12)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    for (logname, records), color in zip(all_records.items(), colors):
        steps = [r.get('total_timesteps', 0) for r in records]
        ev    = [r.get('explained_variance', 0) for r in records]
        vl    = [r.get('value_loss', 0) for r in records]
        pgl   = [r.get('policy_gradient_loss', 0) for r in records]
        ent   = [-r.get('entropy_loss', 0) for r in records]   # negate: loss = -entropy

        label = logname.replace('.txt', '')

        axes[0, 0].plot(steps, ev, color=color, label=label, linewidth=2)
        axes[0, 1].plot(steps, vl, color=color, label=label, linewidth=2)
        axes[1, 0].plot(steps, pgl, color=color, label=label, linewidth=2)
        axes[1, 1].plot(steps, ent, color=color, label=label, linewidth=2)

    axes[0, 0].set_title('Explained Variance (→ 1.0 = perfect value function)')
    axes[0, 0].set_ylabel('EV')
    axes[0, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='EV=0.9 target')

    axes[0, 1].set_title('Value Loss (↓ = better value estimates)')
    axes[0, 1].set_ylabel('Loss')

    axes[1, 0].set_title('Policy Gradient Loss (↓ = policy improving)')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

    axes[1, 1].set_title('Policy Entropy (nats, ↓ = more deterministic policy)')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].axhline(y=8.32, color='r', linestyle='--', alpha=0.5, label='Max entropy (8.32)')

    for ax in axes.flat:
        ax.set_xlabel('Total Timesteps')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    out = 'results/training_curve.png'
    plt.savefig(out, dpi=150)
    print(f"Plot saved: {out}")

    # Print last values per log
    print("\n--- Latest training stats ---")
    for logname, records in all_records.items():
        last = records[-1]
        print(f"\n{logname}:")
        for k in ['total_timesteps', 'fps', 'explained_variance', 'value_loss',
                  'policy_gradient_loss', 'entropy_loss']:
            if k in last:
                print(f"  {k}: {last[k]}")


if __name__ == '__main__':
    main()
