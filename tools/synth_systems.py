"""python tools/synth_systems.py <homr dir> <out dir> <count> <seed>
Pages of staffs grouped into systems the way homr's detection can get them wrong, and
homr's repair_systems on each: a voice and a piano, grand staffs only, quartets of
single staffs, a layout which changes on the page; with grand staffs whose brace was
missed, systems joined by a crease, staffs hidden in some systems, a stray staff at the
end, singles wrongly paired, nothing grouped at all. Written as one json file per case
(the grouped staffs and homr's result, or the error), for the `check_systems` binary.
PARSE=1 also runs homr's parse_staffs on the repaired systems, with the transformer
replaced by random symbols for each staff (recorded, in homr's calling order), for the
`check_parse` binary: which staff is read when, hidden staffs and their measure rests."""
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
    shape = rng.choice(["voice+piano", "voice+piano", "grand", "grand", "singles", "singles", "intro", "choir+piano", "swapped"])
    systems = rng.randint(1, 7)
    kinds = {"voice+piano": [["s", "g"]], "grand": [["g"]], "singles": [["s"] * rng.randint(2, 4)],
             "intro": [["g"]] * 2 + [["s", "g"]] * 5, "choir+piano": [["s", "s", "g"]],
             "swapped": [["s", "g"], ["g", "s"], ["g"]]}[shape]
    y = rng.uniform(60, 120)
    result = []
    for i in range(systems):
        kind = kinds[i % len(kinds)] if shape == "swapped" else kinds[i if shape == "intro" and i < len(kinds) else 0 if shape != "intro" else 2]
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
        row = rows[rng.randrange(len(rows))]
        shifted = Staff([StaffPoint(p.x + 0.4, p.y, p.angle) for p in row[0].grid])
        shifted.is_grandstaff = row[0].is_grandstaff
        row[0] = shifted
    if rng.random() < 0.12 and len(rows) > 2:  # rows not in order down the page (MultiStaff sorts its own staffs)
        i = rng.randrange(len(rows) - 2)
        rows[i:i + 3] = rows[i:i + 3][::-1]
    return [MultiStaff(r, []) for r in rows]


def dump(rows):
    return [[{"grand": s.is_grandstaff, "grid": [[p.x, [float(v) for v in p.y], float(p.angle)] for p in s.grid]} for s in row.staffs] for row in rows]


PARSE = bool(os.environ.get('PARSE'))
import homr.staff_parsing as sp  # noqa: E402
from homr.transformer.vocabulary import EncodedSymbol  # noqa: E402


def ser(v):
    return [{'rhythm': x.rhythm, 'pitch': x.pitch, 'lift': x.lift, 'articulation': x.articulation, 'slur': x.slur,
             'position': x.position} for x in v]


def random_staff_symbols(grand):
    if rng.random() < 0.06:
        return []  # an empty staff
    E, N = '_', '.'
    out = [EncodedSymbol('clef_G2', E, E, E, E, 'upper')]
    if grand:
        out += [EncodedSymbol('chord'), EncodedSymbol('clef_F4', E, E, E, E, 'lower')]
    out.append(EncodedSymbol('keySignature_0'))
    if rng.random() < 0.7:
        out.append(EncodedSymbol('timeSignature/4'))
    for _ in range(rng.randint(1, 4)):
        for _ in range(rng.randint(1, 5)):
            pos = rng.choice(['upper', 'lower'] if grand else ['upper'])
            out.append(EncodedSymbol(rng.choice(['note_4', 'note_8', 'note_2', 'rest_4']), 'C4' if rng.random() < 0.8 else E, E, E, E, pos))
        out.append(EncodedSymbol(rng.choice(['barline', 'barline', 'barline', 'doublebarline', 'repeatEnd'])))
    return out


stats = {"ok": 0, "error": 0, "changed": 0, "regrouped by pattern": 0, "single staff dropped": 0, "every staff on its own": 0, "staffs hidden in some systems": 0}
for n in range(count):
    rows = grouped(page())
    case = {"input": dump(rows)}
    try:
        with contextlib.redirect_stderr(io.StringIO()) as log:
            result = repair_systems(debug, rows)
        case["output"] = dump(result)
        if PARSE:
            calls = []

            def stub(debug_, index, staff_, image, regions, config, page_to_input_image=None):
                symbols = random_staff_symbols(staff_.is_grandstaff)
                calls.append({"index": index, "staff": [staff_.min_x, staff_.max_x, staff_.min_y, staff_.max_y, bool(staff_.is_grandstaff)], "symbols": ser(symbols)})
                return symbols

            sp.parse_staff_image = stub
            for key, systems in (("", result), ("raw_", rows)):  # as repaired, and as detected
                del calls[:]
                case[key + "calls"] = calls[:]
                with contextlib.redirect_stderr(io.StringIO()) as plog:
                    try:
                        case[key + "voices"] = [ser(v) for v in sp.parse_staffs(debug, systems, np.zeros((10, 10), np.uint8), None)]
                    except BaseException as e:  # noqa
                        case[key + "parse_error"] = f"{type(e).__name__}: {e}"
                stats["every staff on its own"] += "reading every staff on its own" in plog.getvalue()
                stats["staffs hidden in some systems"] += "Staffs are hidden in some systems" in plog.getvalue()
                case[key + "calls"] = calls[:]
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
