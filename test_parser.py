#!/usr/bin/env python3
"""Test script to debug the parser."""
from pathlib import Path
from envs.fluid_stats.performance_stats_parser import (
    load_all_performance_records_from_file,
)

if __name__ == "__main__":
    log_path = Path("/home/superfhwl/.orcagym/tmp/orcasph_20260419_211145.log")
    print(f"Loading log file: {log_path}")
    print(f"File exists: {log_path.exists()}")
    
    records = load_all_performance_records_from_file(log_path)
    print(f"Loaded {len(records)} performance records")
    
    if records:
        print(f"\nFirst record: {records[0]}")
