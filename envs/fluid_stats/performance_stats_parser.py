#!/usr/bin/env python3
"""
Parse OrcaSPH performance stats lines of the form::

    [TIME_STATS] initializeCPUDataEachTimeStep=1.234 initializeGPUDataEachTimeStep=0.567 ...

Used by the matplotlib performance-stats viewer; kept free of pyplot and gymnasium.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STAT_MARKER = "[TIME_STATS]"


def parse_performance_stats_line(line: str) -> Optional[Dict[str, Any]]:
    """Return a dict of typed fields, or None if the line is not a stats line."""
    if STAT_MARKER not in line:
        return None
    i = line.find(STAT_MARKER)
    rest = line[i + len(STAT_MARKER) :].strip()
    if not rest:
        return None
    parts = rest.split()
    raw: Dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        raw[k] = v
    if not raw:
        return None
    try:
        out: Dict[str, Any] = {}
        for k, v in raw.items():
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        return out
    except (KeyError, ValueError):
        return None


@dataclass
class TailState:
    """Incremental read state for a single log file."""

    path: Path
    offset: int = 0


def read_new_performance_records(
    path: Path,
    state: TailState,
) -> List[Dict[str, Any]]:
    """Read newly appended bytes since *state.offset* and return parsed performance records."""
    state.path = path
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size < state.offset:
        state.offset = 0
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(state.offset)
            chunk = f.read()
            state.offset = f.tell()
    except OSError:
        return []
    
    # Parse multi-line [TIME_STATS] blocks
    in_stats_block = False
    current_record = {}
    
    for line in chunk.splitlines():
        line = line.strip()
        if STAT_MARKER in line:
            # Start of a new stats block
            if in_stats_block and current_record:
                records.append(current_record)
            in_stats_block = True
            current_record = {}
            # Check if there's any key=value on the same line
            rest = line[line.find(STAT_MARKER) + len(STAT_MARKER):].strip()
            if rest and "=" in rest:
                for p in rest.split():
                    if "=" in p:
                        k, v = p.split("=", 1)
                        try:
                            current_record[k] = float(v)
                        except ValueError:
                            current_record[k] = v
        elif in_stats_block:
            if "=" in line:
                # This is a line with key=value
                try:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        current_record[k] = float(v)
                    except ValueError:
                        current_record[k] = v
                except:
                    pass
            else:
                # End of stats block (line without =)
                if current_record:
                    records.append(current_record)
                in_stats_block = False
                current_record = {}
    
    # Add the last record if we're still in a block
    if in_stats_block and current_record:
        records.append(current_record)
    
    return records


def merge_dedupe(accumulator: List[Dict[str, Any]], new_records: List[Dict[str, Any]]) -> None:
    """Append *new_records* to *accumulator*, dropping duplicates."""
    for r in new_records:
        # Check if this record is already in the accumulator
        duplicate = False
        for existing in accumulator:
            if existing == r:
                duplicate = True
                break
        if not duplicate:
            accumulator.append(r)


def compute_average_times(records: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute average times for each component."""
    if not records:
        return {}
    
    # Get all unique keys
    keys = set()
    for record in records:
        keys.update(record.keys())
    
    # Compute averages
    averages = {}
    for key in keys:
        values = []
        for record in records:
            if key in record:
                values.append(float(record[key]))
        if values:
            averages[key] = sum(values) / len(values)
    
    return averages


def load_all_performance_records_from_file(path: Path) -> List[Dict[str, Any]]:
    """Parse every performance stats line in *path* from the beginning."""
    if not path.exists():
        return []
    
    state = TailState(path, offset=0)
    return read_new_performance_records(path, state)