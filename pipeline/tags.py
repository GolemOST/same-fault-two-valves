"""
The 52 recorded variables and 20 faults of the Tennessee Eastman benchmark, with the
equipment each one belongs to.

Source of names, units, fault types and the descriptions of faults 16-20: the revised TE
model code in N. L. Ricker's Tennessee Eastman archive, temexd_mod.zip -> temexd_mod.c,
header comment lines 238-249 (manipulated variables), 251-276 (disturbances) and 294-341
(measured values). https://depts.washington.edu/control/LARRY/TE/download.html (read 2026-10-22).
Where that code says the SEPARATOR's cooling water (inputs/states, IDV 5, 12, 15), we say the
CONDENSER's, as Du & Du (2018) do; the code's own outputs 22/48/49 call the same stream
"condenser cooling water". Faults 16-20 are marked "(unknown)" in the code.

The `equipment` grouping is ours (by stream / unit), for joining notes to sensors.
"""

MEASURED = [  # xmeas_1 .. xmeas_41
    ("Feed flow, A (stream 1)", "kscmh", "feeds"),
    ("Feed flow, D (stream 2)", "kg/h", "feeds"),
    ("Feed flow, E (stream 3)", "kg/h", "feeds"),
    ("Feed flow, A & C (stream 4)", "kscmh", "feeds"),
    ("Recycle flow to reactor (stream 8)", "kscmh", "compressor"),
    ("Reactor feed (stream 6)", "kscmh", "reactor"),
    ("Reactor pressure", "kPa gauge", "reactor"),
    ("Reactor level", "%", "reactor"),
    ("Reactor temperature", "°C", "reactor"),
    ("Purge flow (stream 9)", "kscmh", "purge"),
    ("Separator temperature", "°C", "separator"),
    ("Separator level", "%", "separator"),
    ("Separator pressure", "kPa gauge", "separator"),
    ("Separator underflow", "m³/h", "separator"),
    ("Stripper level", "%", "stripper"),
    ("Stripper pressure", "kPa gauge", "stripper"),
    ("Stripper underflow (stream 11)", "m³/h", "stripper"),
    ("Stripper temperature", "°C", "stripper"),
    ("Stripper steam flow", "kg/h", "stripper"),
    ("Compressor work", "kW", "compressor"),
    ("Reactor cooling water outlet temperature", "°C", "reactor"),
    ("Condenser cooling water outlet temperature", "°C", "condenser"),
] + [(f"Reactor feed analysis, {c} (stream 6)", "mol %", "reactor") for c in "ABCDEF"] \
  + [(f"Purge analysis, {c} (stream 9)", "mol %", "purge") for c in "ABCDEFGH"] \
  + [(f"Product analysis, {c} (stream 11)", "mol %", "stripper") for c in "DEFGH"]

MANIPULATED = [  # xmv_1 .. xmv_11 (agitator speed, XMV 12, is constant and not in the data)
    ("Valve position, D feed (stream 2)", "%", "feeds"),
    ("Valve position, E feed (stream 3)", "%", "feeds"),
    ("Valve position, A feed (stream 1)", "%", "feeds"),
    ("Valve position, A & C feed (stream 4)", "%", "feeds"),
    ("Valve position, compressor recycle", "%", "compressor"),
    ("Valve position, purge (stream 9)", "%", "purge"),
    ("Valve position, separator underflow (stream 10)", "%", "separator"),
    ("Valve position, stripper underflow (stream 11)", "%", "stripper"),
    ("Valve position, stripper steam", "%", "stripper"),
    ("Valve position, reactor cooling water", "%", "reactor"),
    ("Valve position, condenser cooling water", "%", "condenser"),
]

FAULTS = {  # IDV number: (type, what is disturbed)
    1: ("Step", "A/C ratio of stream 4, B composition constant"),
    2: ("Step", "B composition of stream 4, A/C ratio constant"),
    3: ("Step", "D feed (stream 2) temperature"),
    4: ("Step", "Reactor cooling water inlet temperature"),
    5: ("Step", "Condenser cooling water inlet temperature"),
    6: ("Step", "A feed loss (stream 1)"),
    7: ("Step", "C header pressure loss (stream 4)"),
    8: ("Random", "A/B/C composition of stream 4"),
    9: ("Random", "D feed (stream 2) temperature"),
    10: ("Random", "C feed (stream 4) temperature"),
    11: ("Random", "Reactor cooling water inlet temperature"),
    12: ("Random", "Condenser cooling water inlet temperature"),
    13: ("Drift", "Reaction kinetics"),
    14: ("Stiction", "Reactor cooling water valve"),
    15: ("Stiction", "Condenser cooling water valve"),
    16: ("Random", "(unknown) Heat transfer within the stripper heat exchanger"),
    17: ("Random", "(unknown) Heat transfer within the reactor"),
    18: ("Random", "(unknown) Heat transfer within the condenser"),
    19: ("Stiction", "(unknown) Compressor recycle, separator and stripper underflow, stripper steam valves"),
    20: ("Random", "(unknown)"),
}


def catalogue() -> list[dict]:
    """All 52 variables in data-column order, as dicts."""
    rows = []
    for index, (name, unit, equipment) in enumerate(MEASURED, start=1):
        rows.append({"column": f"xmeas_{index}", "tag": f"XMEAS({index})", "name": name,
                     "unit": unit, "equipment": equipment, "kind": "measured"})
    for index, (name, unit, equipment) in enumerate(MANIPULATED, start=1):
        rows.append({"column": f"xmv_{index}", "tag": f"XMV({index})", "name": name,
                     "unit": unit, "equipment": equipment, "kind": "manipulated"})
    return rows
