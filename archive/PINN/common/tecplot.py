"""Minimal DEVSIM Tecplot reader for node-centered electrostatic fields."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _cell_centered_indices(header: str) -> set[int]:
    match = re.search(r"VARLOCATION=\(\[([^\]]+)\]=CELLCENTERED\)", header)
    if match is None:
        return set()
    indices: set[int] = set()
    for part in match.group(1).split(","):
        part = part.strip()
        if "-" in part:
            lower, upper = (int(value) for value in part.split("-", 1))
            indices.update(range(lower, upper + 1))
        else:
            indices.add(int(part))
    return indices


def read_devsim_tecplot(path: str | Path) -> dict[str, np.ndarray]:
    """Read the largest node-centered zone from a DEVSIM Tecplot file."""

    path = Path(path)
    text = path.read_text()
    variable_match = re.search(r"VARIABLES\s*=\s*(.+?)\nZONE", text, flags=re.S)
    if variable_match is None:
        raise ValueError(f"could not find VARIABLES header in {path}")
    variables = re.findall(r'"([^"]+)"', variable_match.group(1))
    zones = list(re.finditer(r"(?m)^ZONE\b.*$", text))
    wanted = {"x", "y", "z", "Potential", "Potential_gradx", "Potential_grady", "Potential_gradz"}
    best: dict[str, np.ndarray] | None = None
    for zone_index, zone in enumerate(zones):
        header = zone.group(0)
        node_match = re.search(r"NODES=(\d+)", header)
        element_match = re.search(r"ELEMENTS=(\d+)", header)
        if node_match is None or element_match is None:
            continue
        node_count = int(node_match.group(1))
        element_count = int(element_match.group(1))
        end = zones[zone_index + 1].start() if zone_index + 1 < len(zones) else len(text)
        tokens = np.fromstring(text[zone.end():end], sep=" ")
        cell_centered = _cell_centered_indices(header)
        arrays: dict[str, np.ndarray] = {}
        offset = 0
        for variable_index, variable_name in enumerate(variables, start=1):
            count = element_count if variable_index in cell_centered else node_count
            if offset + count > len(tokens):
                break
            values = tokens[offset:offset + count]
            offset += count
            if variable_name in wanted and count == node_count:
                arrays[variable_name] = values
        if {"x", "y", "Potential"}.issubset(arrays):
            if best is None or len(arrays["x"]) > len(best["x"]):
                best = arrays
    if best is None:
        raise ValueError(f"could not find a node-centered electrostatic zone in {path}")
    return best
