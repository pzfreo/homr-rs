"""python tools/synth_systems.py <homr dir> <out dir> <count> <seed>
Pages of staffs grouped into systems the way homr's detection can get them wrong, and
homr's repair_systems on each: a voice and a piano, grand staffs only, quartets of
single staffs, a layout which changes on the page; with grand staffs whose brace was
missed, systems joined by a crease, staffs hidden in some systems, a stray staff at the
end, singles wrongly paired, nothing grouped at all. Written as one json file per case
(the grouped staffs and homr's result, or the error), for the `check_systems` binary."""
import contextlib, io, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from homr_setup import load

load(os.path.abspath(sys.argv[1]))
from homr.debug import Debug  # noqa: E402
from homr.model import MultiStaff, Staff, StaffPoint  # noqa: E402
from homr.system_repair import repair_systems  # noqa: E402

out, count, seed = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
rng = random.Random(seed)
debug = Debug(np.zeros((10, 10, 3), np.uint8), "synthetic.png", False)


def staff(y0, unit, x0, x1, tilt, jitter_x=0.0):
    grid = [StaffPoint(float(x) + jitter_x, [y0 + i * unit + tilt * (x - x0) for i in range(5)], 0.0) for x in range(x0, x1 + 1, 10)]
    return Staff(grid)


def page():
    """The staffs of the page, top to bottom: ('single', Staff) or ('grand', upper, lower), per true system."""
    unit = rng.uniform(9, 15)
    x0, x1 = rng.randrange(80, 140, 10), rng.randrange(900, 1100, 10)
    tilt = rng.uniform(-0.01, 0.01)
    shape = rng.choice(["voice+piano", "voice+piano", "grand", "grand", "singles", "singles", "intro", "choir+piano"])
    systems = rng.randint(1, 7)
    kinds = {"voice+piano": [["s", "g"]], "grand": [["g"]], "singles": [["s"] * rng.randint(2, 4)],
             "intro": [["g"]] * 2 + [["s", "g"]] * 5, "choir+piano": [["s", "s", "g"]]}[shape]
    y = rng.uniform(60, 120)
    result = []
    for i in range(systems):
        kind = kinds[i if shape == "intro" and i < len(kinds) else 0 if shape != "intro" else 2]
        row = []
        for k in kind:
            xa, xb = x0 + rng.choice([0, 0, 0, 10]), x1 - rng.choice([0, 0, 0, 10])
            if k == "s":
                row.append(("single", staff(y, unit, xa, xb, tilt)))
                y += 4 * unit + rng.uniform(5, 7) * unit
            else:
                gap = rng.uniform(5.5, 8) * unit
                row.append(("grand", staff(y, unit, xa, xb, tilt), staff(y + 4 * unit + gap, unit, xa, xb, tilt)))
                y += 8 * unit + gap + rng.uniform(5, 7) * unit
        result.append(row)
        y += rng.uniform(1, 4) * unit
    return result


def grouped(true_systems):
    """MultiStaffs as detection could give them, with defects."""
    rows = []
    defect_rate = rng.choice([0.0, 0.15, 0.4, 0.8])
    for system in true_systems:
        row = []
        for item in system:
            if item[0] == "single":
                row.append(item[1])
            elif rng.random() < defect_rate:  # the brace was missed: two single staffs
                if rng.random() < 0.25:  # which don't overlap: each covers one side of the page
                    mid = (item[1].min_x + item[1].max_x) / 2
                    upper = Staff([p for p in item[1].grid if p.x <= mid - 20])
                    lower = Staff([p for p in item[2].grid if p.x > mid + 20])
                    row += [upper, lower]
                else:
                    row += [item[1], item[2]]
            else:
                merged = item[1].merge(item[2])
                row.append(merged)
        if rng.random() < defect_rate / 2 and len(row) > 1:  # hide a staff in this system
            row.pop(0)
        rows.append(row)
    if rng.random() < defect_rate and len(rows) > 2:  # a crease joins two systems
        i = rng.randrange(len(rows) - 1)
        rows[i:i + 2] = [rows[i] + rows[i + 1]]
    if rng.random() < 0.5 * defect_rate or rng.random() < 0.08:  # nothing grouped
        rows = [[s] for row in rows for s in row]
    if rng.random() < 0.2:  # a stray single staff below the last system
        last = max((s.max_y for row in rows for s in row), default=100)
        rows.append([staff(last + rng.uniform(1, 6) * 12, 12, 100, rng.choice([300, 600, 900]), 0.0)])
    if rng.random() < 0.05:  # points a little off the common positions
        s = rows[rng.randrange(len(rows))][0]
        s.grid = [StaffPoint(p.x + 0.4, p.y, p.angle) for p in s.grid]
    if rng.random() < 0.12 and len(rows) > 2:  # rows not in order down the page (MultiStaff sorts its own staffs)
        i = rng.randrange(len(rows) - 2)
        rows[i:i + 3] = rows[i:i + 3][::-1]
    return [MultiStaff(r, []) for r in rows]


def dump(rows):
    return [[{"grand": s.is_grandstaff, "grid": [[p.x, [float(v) for v in p.y], float(p.angle)] for p in s.grid]} for s in row.staffs] for row in rows]


stats = {"ok": 0, "error": 0, "changed": 0, "regrouped by pattern": 0, "single staff dropped": 0}
for n in range(count):
    rows = grouped(page())
    case = {"input": dump(rows)}
    try:
        with contextlib.redirect_stderr(io.StringIO()) as log:
            result = repair_systems(debug, rows)
        case["output"] = dump(result)
        case["log"] = log.getvalue()
        stats["ok"] += 1
        stats["changed"] += case["output"] != case["input"]
        stats["regrouped by pattern"] += "grouping them by this pattern" in case["log"]
        stats["single staff dropped"] += "Removing a single staff" in case["log"]
    except BaseException as e:  # noqa
        case["error"] = f"{type(e).__name__}: {e}"
        stats["error"] += 1
    json.dump(case, open(os.path.join(out, f"systems-{seed}-{n}.json"), "w"))
print(stats)
