#!/usr/bin/env python3
"""Test script to visualize log file with static plot."""
from pathlib import Path
import matplotlib.pyplot as plt

from envs.fluid_stats.performance_stats_parser import (
    load_all_performance_records_from_file,
)

if __name__ == "__main__":
    log_path = Path("/home/superfhwl/.orcagym/tmp/orcasph_20260419_211145.log")
    print(f"Loading log file: {log_path}")
    
    records = load_all_performance_records_from_file(log_path)
    print(f"Loaded {len(records)} performance records")
    
    if not records:
        print("No records found!")
        exit(1)
    
    print(f"\nFirst record: {records[0]}")
    
    # Setup matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("OrcaSPH Performance Stats")
    
    # Get all unique keys
    keys = set()
    for record in records:
        keys.update(record.keys())
    
    print(f"\nFound {len(keys)} unique keys: {sorted(keys)}")
    
    # Plot each key
    for key in sorted(keys):
        values = []
        for record in records:
            if key in record:
                values.append(record[key])
            else:
                values.append(0.0)
        print(f"Plotting {key}: min={min(values):.3f}, max={max(values):.3f}, avg={sum(values)/len(values):.3f}")
        ax.plot(range(len(values)), values, label=key)
    
    ax.set_title("OrcaSPH Performance Stats")
    ax.set_xlabel("Step")
    ax.set_ylabel("Time (ms)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    
    print("\nShowing plot...")
    plt.show()
