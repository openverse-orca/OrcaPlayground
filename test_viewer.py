#!/usr/bin/env python3
"""Test script to directly visualize the log file."""
from pathlib import Path
import sys
import time

from envs.fluid_stats.performance_stats_viewer import main as viewer_main

if __name__ == "__main__":
    log_path = Path("/home/superfhwl/.orcagym/tmp/orcasph_20260419_211145.log")
    # Override sys.argv to simulate command-line call
    sys.argv = [sys.argv[0], str(log_path), "--interval", "0.5", "--rolling", "30"]
    
    print("Starting viewer...")
    print(f"Log file: {log_path}")
    viewer_main()
