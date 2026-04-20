#!/usr/bin/env python3
"""
Real-time viewer for OrcaSPH performance stats using matplotlib bar chart.

Usage:
    python performance_stats_viewer.py <log_file>
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from performance_stats_parser import (
    TailState,
    read_new_performance_records,
    merge_dedupe,
)


def group_keys_by_category(keys: List[str]) -> Dict[str, List[str]]:
    """Group performance keys by categories for better visualization."""
    categories = {
        "DFSPH GPU": [
            "computeVolumeAndBoundaryXCUDA",
            "computeDensitiesCUDA",
            "computeDFSPHFactorCUDA",
            "divergenceSolveCUDA",
            "pressureSolveCUDA",
            "computeNonPressureForcesCUDA",
            "compute_new_velocitiesCUDA",
            "computeFinalPositionsCUDA",
            "clearAccelerationsCUDA",
            "updateTimeStepSizeCUDA",
            "precomputeValues",
        ],
        "Neighborhood Search": [
            "neighborhood_search",
            "reformatNeighborDataFromCuNSearch",
            "initializeNeighborIndicesGPUEachTimeStep",
        ],
        "Data Transfer (PCIe)": [
            "EachtimeStepInH2D",
            "EachtimeStepOutD2H",
        ],
        "Init & Others": [
            "initializeCPUDataEachTimeStep",
            "initializeGPUDataEachTimeStep",
            "emitParticles",
            "animateParticles",
        ],
        "PBD": [
            "SimStep_-_PBD",
            "simulation_step",
            "collision_detection",
            "position_constraints_projection",
        ],
    }
    
    # Build reverse mapping
    key_to_category = {}
    for cat, keys_list in categories.items():
        for key in keys_list:
            key_to_category[key] = cat
    
    # Group
    result = defaultdict(list)
    uncategorized = []
    for key in keys:
        cat = key_to_category.get(key)
        if cat:
            result[cat].append(key)
        else:
            uncategorized.append(key)
    
    if uncategorized:
        result["Other"] = uncategorized
    
    return result


def plot_bar_chart(
    ax: plt.Axes,
    data: Dict[str, float],
    title: str,
    xlabel: str,
    ylabel: str,
    group_by_category: bool = True,
) -> None:
    """Plot a bar chart on given axes."""
    if not data:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
        return
    
    if group_by_category:
        categories = group_keys_by_category(list(data.keys()))
        # Flatten ordered by category
        ordered_keys = []
        for cat in ["DFSPH GPU", "Neighborhood Search", "Data Transfer (PCIe)", "Init & Others", "PBD", "Other"]:
            if cat in categories:
                ordered_keys.extend(categories[cat])
    else:
        # Sort by value descending
        ordered_keys = sorted(data.keys(), key=lambda k: -data[k])
    
    # Filter out keys not in data
    ordered_keys = [k for k in ordered_keys if k in data]
    if not ordered_keys:
        ordered_keys = list(data.keys())
    
    values = [data[k] for k in ordered_keys]
    
    # Plot
    colors = plt.cm.tab20(np.linspace(0, 1, len(ordered_keys)))
    bars = ax.bar(ordered_keys, values, color=colors)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha='center',
                va='bottom',
                fontsize=8,
                rotation=0
            )
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    ax.grid(True, axis='y', alpha=0.3)


def compute_cumulative_average(
    all_records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute cumulative average across all records."""
    if not all_records:
        return {}
    
    # Sum all values
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    
    for record in all_records:
        for k, v in record.items():
            if isinstance(v, (int, float)):
                sums[k] += v
                counts[k] += 1
    
    # Compute averages
    return {k: sums[k] / counts[k] for k in sums}


def main():
    parser = argparse.ArgumentParser(description="Real-time viewer for OrcaSPH performance stats")
    parser.add_argument("log_file", type=Path, help="Path to OrcaSPH log file")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")
    parser.add_argument("--rolling", type=int, default=10, help="Maximum number of data points to keep for rolling average")
    args = parser.parse_args()

    print(f"[DEBUG] Starting viewer with log file: {args.log_file}")
    print(f"[DEBUG] Update interval: {args.interval}s")

    # Initialize state
    state = TailState(args.log_file)
    all_records: List[Dict[str, Any]] = []
    cumulative_dt_stats: Dict[float, int] = defaultdict(int)

    # Setup matplotlib
    plt.ion()
    fig = plt.figure(figsize=(16, 8))
    fig.canvas.manager.set_window_title("OrcaSPH Performance Stats")

    ax = fig.add_subplot(111)

    ax.set_title("Waiting for data...")
    ax.set_xlabel("Components")
    ax.set_ylabel("Time (ms)")
    fig.canvas.draw()
    fig.canvas.flush_events()

    try:
        while True:
            new_records = read_new_performance_records(args.log_file, state)
            print(f"[DEBUG] Read {len(new_records)} new records, total records now: {len(all_records) + len(new_records)}")
            
            if new_records:
                cleaned_new_records = []
                for record in new_records:
                    cleaned = {k: v for k, v in record.items() if k != "_dt_s"}
                    cleaned_new_records.append(cleaned)
                    
                    if "_dt_s" in record:
                        dt = record["_dt_s"]
                        cumulative_dt_stats[dt] += 1
                
                merge_dedupe(all_records, cleaned_new_records)
                
                if len(all_records) > args.rolling:
                    all_records = all_records[-args.rolling:]
                
                print(f"[DEBUG] Total records: {len(all_records)}")
                
                ax.clear()
                if all_records:
                    rolling_avg = compute_cumulative_average(all_records)
                    
                    dt_info = ""
                    if cumulative_dt_stats:
                        dt_items = sorted(cumulative_dt_stats.items())
                        if len(dt_items) == 1:
                            dt, cnt = dt_items[0]
                            dt_info = f" | Timestep: {dt*1000:.2f} ms ({cnt} batches)"
                        else:
                            dt_info = " | Timesteps: " + ", ".join([f"{dt*1000:.2f}ms x{cnt}" for dt, cnt in dt_items])
                    
                    plot_bar_chart(
                        ax,
                        rolling_avg,
                        title=f"Rolling Average (last {len(all_records)} batches){dt_info}",
                        xlabel="Components",
                        ylabel="Time (ms)",
                        group_by_category=True,
                    )
                    print(f"[DEBUG] Updated chart with rolling average")
                else:
                    ax.set_title("Waiting for data...")
                    ax.set_xlabel("Components")
                    ax.set_ylabel("Time (ms)")
                
                fig.subplots_adjust(bottom=0.15)
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])
                fig.canvas.draw_idle()
            
            fig.canvas.flush_events()
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