#!/usr/bin/env python3
"""
Real-time viewer for OrcaSPH performance stats using matplotlib.

Usage:
    python performance_stats_viewer.py <log_file>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from performance_stats_parser import (
    TailState,
    read_new_performance_records,
    merge_dedupe,
    compute_average_times,
)


def main():
    parser = argparse.ArgumentParser(description="Real-time viewer for OrcaSPH performance stats")
    parser.add_argument("log_file", type=Path, help="Path to OrcaSPH log file")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    parser.add_argument("--rolling", type=int, default=50, help="Maximum number of data points to display")
    args = parser.parse_args()

    print(f"[DEBUG] Starting viewer with log file: {args.log_file}")
    print(f"[DEBUG] Update interval: {args.interval}s, Rolling window: {args.rolling}")

    # Initialize state
    state = TailState(args.log_file)
    records: List[Dict[str, Any]] = []
    step = 0

    # Setup matplotlib
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title("OrcaSPH Performance Stats")
    lines: Dict[str, Any] = {}
    labels: List[str] = []

    try:
        while True:
            # Read new records
            new_records = read_new_performance_records(args.log_file, state)
            print(f"[DEBUG] Read {len(new_records)} new records, total records now: {len(records) + len(new_records)}")
            
            if new_records:
                merge_dedupe(records, new_records)
                print(f"[DEBUG] After merge, total records: {len(records)}")
                
                # Limit to rolling window
                if len(records) > args.rolling:
                    records = records[-args.rolling:]
                    print(f"[DEBUG] After rolling window, records: {len(records)}")
                
                # Update plot
                ax.clear()
                if records:
                    print(f"[DEBUG] Updating plot with {len(records)} records")
                    # Get all unique keys
                    keys = set()
                    for record in records:
                        keys.update(record.keys())
                    
                    print(f"[DEBUG] Found {len(keys)} unique keys: {sorted(keys)}")
                    
                    # Plot each key
                    for key in keys:
                        values = []
                        for record in records:
                            if key in record:
                                values.append(record[key])
                            else:
                                values.append(0.0)
                        print(f"[DEBUG] Plotting {key}: min={min(values):.3f}, max={max(values):.3f}, avg={sum(values)/len(values):.3f}")
                        ax.plot(range(len(values)), values, label=key)
                    
                    ax.set_title("OrcaSPH Performance Stats")
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Time (ms)")
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    fig.tight_layout()
                    plt.draw()
                    print(f"[DEBUG] Plot updated and drawn")
            
            # Pause and sleep
            plt.pause(0.1)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close()


if __name__ == "__main__":
    main()