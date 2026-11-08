"""python tools/synth_photos.py <score page images dir> <out dir> <count> <seed>
Photo-like images for testing autocrop: a score page (taken from the given directory)
laid on a desk, rotated and in perspective, with a shadow gradient and a lamp's color
cast, JPEG noise; sometimes off-centre, filling the frame, tiny, in a book (two pages,
a dark gutter) or on a desk which is nearly the paper's color."""
import glob, os, random, sys
import cv2
import numpy as np

src, out, count, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
rng = random.Random(seed)
# the score pages of the corpus, not its screenshots of the app (named by what they show)
SKIP_PREFIXES = ("t-", "app", "lib", "home", "shot")
pages = [p for p in sorted(glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.jpg")))
         if not os.path.basename(p).startswith(SKIP_PREFIXES) and "teaser" not in os.path.basename(p)]
if not pages:
    sys.exit("no score pages in " + src)


def desk(h, w):
    base = np.array([rng.randint(20, 235) for _ in range(3)], np.float32)
    if rng.random() < 0.25:  # a white table: nearly the paper's color
        base = np.array([rng.randint(185, 215)] * 3, np.float32) + np.array([rng.randint(-8, 8) for _ in range(3)], np.float32)
    low = cv2.resize(np.random.default_rng(rng.randint(0, 10**6)).random((4, 4, 1), dtype=np.float32), (w, h), interpolation=cv2.INTER_CUBIC)[:, :, None]
    img = base[None, None, :] * (0.75 + 0.5 * low)
    if rng.random() < 0.5:  # wood grain
        grain = np.sin(np.arange(w, dtype=np.float32) / rng.uniform(3, 12))[None, :, None] * rng.uniform(2, 14)
        img = img + grain
    return np.clip(img + np.random.default_rng(rng.randint(0, 10**6)).normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)


def paper(path, width):
    page = cv2.imread(path)
    s = width / page.shape[1]
    page = cv2.resize(page, (width, int(page.shape[0] * s)))
    h, w = page.shape[:2]
    deep = rng.random() < 0.3  # one side in deep shadow: darker than autocrop's threshold, same color
    gx = np.linspace(rng.uniform(0.3, 0.55) if deep else rng.uniform(0.6, 1.0), rng.uniform(0.85, 1.05), w, dtype=np.float32)[None, :, None]
    if deep and rng.random() < 0.5:
        gx = gx[:, ::-1]
    gy = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.7, 1.05), h, dtype=np.float32)[:, None, None]
    cast = np.array([rng.uniform(0.85, 1.05), rng.uniform(0.92, 1.05), rng.uniform(0.95, 1.1)], np.float32)  # warm lamp (BGR)
    return np.clip(page.astype(np.float32) * gx * gy * cast, 0, 255).astype(np.uint8)


for n in range(count):
    h, w = (rng.randint(260, 480), rng.randint(220, 460)) if rng.random() < 0.25 else (rng.randint(1100, 1800), rng.randint(900, 1500))  # some small
    canvas = desk(h, w)
    mode = rng.choice(["normal", "normal", "off-centre", "fills", "tiny", "book", "shifted"])
    for part in range(2 if mode == "book" else 1):
        pw = {"normal": 0.7, "off-centre": 0.55, "fills": 0.97, "tiny": 0.2, "book": 0.46, "shifted": 0.6}[mode] * w
        pg = paper(rng.choice(pages), int(pw))
        ph, pw_ = pg.shape[:2]
        cx = {"normal": w * 0.5, "off-centre": w * rng.choice([0.3, 0.7]), "fills": w * 0.5, "tiny": w * rng.uniform(0.3, 0.7),
              "book": w * (0.27 if part == 0 else 0.73), "shifted": w * rng.uniform(0.35, 0.65)}[mode] + rng.uniform(-30, 30)
        cy = h * (0.5 if mode != "off-centre" else rng.choice([0.35, 0.65])) + rng.uniform(-40, 40)
        angle = rng.uniform(-14, 14) if mode != "fills" else rng.uniform(-3, 3)
        src_q = np.float32([[0, 0], [pw_, 0], [pw_, ph], [0, ph]])
        rot = cv2.getRotationMatrix2D((pw_ / 2, ph / 2), angle, 1.0)
        dst_q = cv2.transform(src_q[None], rot)[0] + np.float32([cx - pw_ / 2, cy - ph / 2])
        dst_q += np.float32([[rng.uniform(-25, 25), rng.uniform(-25, 25)] for _ in range(4)])  # perspective
        m = cv2.getPerspectiveTransform(src_q, dst_q.astype(np.float32))
        warped = cv2.warpPerspective(pg, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full(pg.shape[:2], 255, np.uint8), m, (w, h), flags=cv2.INTER_LINEAR)
        canvas = np.where(mask[:, :, None] > 127, warped, canvas)
    if mode == "book":  # a dark gutter between the pages
        cv2.line(canvas, (int(w * 0.5), 0), (int(w * 0.5), h), (30, 30, 30), rng.randint(6, 20))
    ok, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
    cv2.imwrite(os.path.join(out, f"photo-{seed}-{n}.png"), cv2.imdecode(enc, cv2.IMREAD_COLOR))
print("made", count)
