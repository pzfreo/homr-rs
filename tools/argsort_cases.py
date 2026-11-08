"""python tools/argsort_cases.py <out dir>
Arrays with many equal values (the case where np.argsort's order is not obvious), and
numpy's argsort of each: <out dir>/cases.json and native.json, for `check_argsort`."""
import json, os, random, sys
import numpy as np

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = random.Random(5)
cases = []
for n in list(range(2, 40)) + [64, 100, 200, 300, 500, 1000, 2000]:
    for kind in range(12):
        k = rng.choice([2, 3, 5, 10, 40, 1000])
        v = [float(rng.randrange(k)) for _ in range(n)]
        if kind % 4 == 1:
            v = sorted(v)
        if kind % 4 == 2:
            v = sorted(v, reverse=True)
        if kind % 6 == 3:
            v = [x + rng.choice([0, 0.5]) for x in v]
        cases.append(v)
json.dump({"values": cases}, open(os.path.join(out, "cases.json"), "w"))
json.dump({"numpy": np.__version__, "order": [np.argsort(np.array(c)).tolist() for c in cases]},
          open(os.path.join(out, "native.json"), "w"))
print(len(cases), "cases, numpy", np.__version__)
