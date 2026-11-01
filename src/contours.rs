//! cv::findContours(image, RETR_TREE, CHAIN_APPROX_SIMPLE), ported from
//! OpenCV 5.0.0 (modules/imgproc/src/contours_new.cpp, contours_common.*)
//! so that contours come out in OpenCV's order with OpenCV's points.

const CHAIN: [(isize, isize); 8] = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)];
const MASK8_RIGHT: i8 = -128; // 0x80
const MASK8_FLAGS: i32 = -2; // 0xFE as schar
const MASK8_LVAL: i32 = 0x7F;
const MAX_SIZE: i32 = 16;

struct Node {
    parent: i32,
    first_child: i32,
    prev: i32,
    next: i32,
    ctable_next: i32,
    pts: Vec<[i32; 2]>,
    brect: (i32, i32, i32, i32), // x, y, width, height
    origin: (i32, i32),
    is_hole: bool,
}

impl Node {
    fn new() -> Node {
        Node { parent: -1, first_child: -1, prev: -1, next: -1, ctable_next: -1, pts: vec![], brect: (0, 0, 0, 0), origin: (0, 0), is_hole: false }
    }
}

struct Scanner {
    external: bool,
    img: Vec<i8>,
    w: usize, // padded width
    h: usize,
    pt: (i32, i32),
    lnbd: (i32, i32),
    nbd: i8,
    tree: Vec<Node>,
    ctable: [i32; 128],
}

const OFFSET: (i32, i32) = (-1, -1);

impl Scanner {
    fn delta(&self, s: i32) -> isize {
        let (dx, dy) = CHAIN[(s % 8) as usize];
        dx + dy * self.w as isize
    }
    fn at(&self, x: i32, y: i32) -> i8 {
        self.img[y as usize * self.w + x as usize]
    }

    fn fetch_contour(&mut self, start: (i32, i32), nbd: i8, node: usize) {
        let i0 = (start.1 as usize * self.w + start.0 as usize) as isize;
        let mut pt = self.tree[node].origin;
        let mut rect = (pt.0, pt.1, pt.0, pt.1);
        let mut s_end: i32 = if self.tree[node].is_hole { 0 } else { 4 };
        let mut s = s_end;
        let mut i1;
        loop {
            s = (s - 1) & 7;
            i1 = i0 + self.delta(s);
            if self.img[i1 as usize] != 0 || s == s_end {
                break;
            }
        }
        if s == s_end {
            self.img[i0 as usize] = nbd | MASK8_RIGHT;
            self.tree[node].pts.push([pt.0, pt.1]);
        } else {
            let mut i3 = i0;
            let mut i4;
            let mut prev_s = s ^ 4;
            loop {
                s_end = s;
                s = s.min(15);
                loop {
                    s += 1;
                    i4 = i3 + self.delta(s);
                    if self.img[i4 as usize] != 0 || s >= MAX_SIZE - 1 {
                        break;
                    }
                }
                s &= 7;
                if ((s - 1) as u32) < (s_end as u32) {
                    self.img[i3 as usize] = nbd | MASK8_RIGHT;
                } else if self.img[i3 as usize] == 1 {
                    self.img[i3 as usize] = nbd;
                }
                if s != prev_s {
                    self.tree[node].pts.push([pt.0, pt.1]);
                    if pt.0 < rect.0 {
                        rect.0 = pt.0;
                    } else if pt.0 > rect.2 {
                        rect.2 = pt.0;
                    }
                    if pt.1 < rect.1 {
                        rect.1 = pt.1;
                    } else if pt.1 > rect.3 {
                        rect.3 = pt.1;
                    }
                }
                prev_s = s;
                pt = (pt.0 + CHAIN[s as usize].0 as i32, pt.1 + CHAIN[s as usize].1 as i32);
                if i4 == i0 && i3 == i1 {
                    break;
                }
                i3 = i4;
                s = (s + 4) & 7;
            }
        }
        rect.2 -= rect.0 - 1;
        rect.3 -= rect.1 - 1;
        self.tree[node].brect = rect;
    }

    fn trace_contour(&self, start: (i32, i32), end: (i32, i32), is_hole: bool) -> bool {
        let stop = (end.1 as usize * self.w + end.0 as usize) as isize;
        let i0 = (start.1 as usize * self.w + start.0 as usize) as isize;
        let s_end: i32 = if is_hole { 0 } else { 4 };
        let mut s = s_end;
        let mut i1;
        loop {
            s = (s - 1) & 7;
            i1 = i0 + self.delta(s);
            if self.img[i1 as usize] != 0 || s == s_end {
                break;
            }
        }
        let mut i3 = i0;
        if s != s_end {
            loop {
                s = s.min(15);
                let mut i4 = i3;
                while s < MAX_SIZE - 1 {
                    s += 1;
                    i4 = i3 + self.delta(s);
                    if self.img[i4 as usize] != 0 {
                        break;
                    }
                }
                if i3 == stop {
                    if self.img[i3 as usize] & MASK8_RIGHT == 0 {
                        return true;
                    }
                    let mut t = s;
                    loop {
                        t = (t - 1) & 7;
                        let i5 = i3 + self.delta(t);
                        if self.img[i5 as usize] != 0 {
                            break;
                        }
                        if t == 0 {
                            return true;
                        }
                    }
                }
                if i4 == i0 && i3 == i1 {
                    break;
                }
                i3 = i4;
                s = (s + 4) & 7;
            }
        } else {
            return i3 == stop;
        }
        false
    }

    fn find_first_bounding_contour(&self, last_pos: (i32, i32), y: i32, lval: i32, par: i32) -> i32 {
        let end = (last_pos.0, y);
        let mut res = par;
        let mut cur = self.ctable[lval as usize];
        while cur != -1 {
            let e = &self.tree[cur as usize];
            if (last_pos.0 - e.brect.0) < e.brect.2 && (last_pos.1 - e.brect.1) < e.brect.3 {
                if res != -1 {
                    let r = &self.tree[res as usize];
                    if self.trace_contour(r.origin, end, r.is_hole) {
                        break;
                    }
                }
                res = cur;
            }
            cur = e.ctable_next;
        }
        res
    }

    fn add_child(&mut self, parent: usize, child: usize) {
        let fc = self.tree[parent].first_child;
        if fc != -1 {
            self.tree[fc as usize].prev = child as i32;
            self.tree[child].next = fc;
        }
        self.tree[parent].first_child = child as i32;
        self.tree[child].parent = parent as i32;
        self.tree[child].prev = -1;
    }

    fn make_contour(&mut self, nbd: &mut i8, is_hole: bool, x: i32, y: i32) -> usize {
        let start = (x - is_hole as i32, y);
        let idx = self.tree.len();
        let mut n = Node::new();
        n.is_hole = is_hole;
        n.origin = (start.0 + OFFSET.0, start.1 + OFFSET.1);
        self.tree.push(n);
        if self.external {
            // isSimple(): marked with MASK8_NEW, no parent table
            self.fetch_contour(start, 2, idx);
            self.tree[idx].origin = start;
            return idx;
        }
        let lval = *nbd;
        *nbd = (((*nbd as i32) + 1) & MASK8_LVAL) as i8;
        if *nbd == 0 {
            *nbd = 1 | 2;
        }
        self.fetch_contour(start, lval, idx);
        self.tree[idx].brect.0 -= OFFSET.0;
        self.tree[idx].brect.1 -= OFFSET.1;
        self.tree[idx].ctable_next = self.ctable[lval as usize];
        self.ctable[lval as usize] = idx as i32;
        self.tree[idx].origin = start;
        idx
    }

    fn contour_scan(&mut self, prev: i32, p: i32, last_pos: &mut (i32, i32), x: i32, y: i32) -> bool {
        let mut is_hole = false;
        if !(prev == 0 && p == 1) {
            if p != 0 || prev < 1 {
                return false;
            }
            if prev & MASK8_FLAGS != 0 {
                last_pos.0 = x - 1;
            }
            is_hole = true;
        }
        if self.external && (is_hole || self.at(last_pos.0, last_pos.1) > 0) {
            return false;
        }
        let mut main_parent: i32 = -1;
        if self.external || last_pos.0 <= 0 {
            main_parent = 0;
        } else {
            let lval = (self.at(last_pos.0, last_pos.1) as i32) & MASK8_LVAL;
            main_parent = self.find_first_bounding_contour(*last_pos, y, lval, main_parent);
            let e = &self.tree[main_parent as usize];
            if e.is_hole == is_hole {
                main_parent = if e.parent != -1 { e.parent } else { 0 };
            }
        }
        last_pos.0 = x - is_hole as i32;
        let mut nbd = self.nbd;
        let c = self.make_contour(&mut nbd, is_hole, x, y);
        if self.tree[c].parent == -1 {
            self.add_child(main_parent as usize, c);
        }
        self.pt = (x + 1, y);
        self.nbd = nbd;
        true
    }

    fn find_next(&mut self) -> bool {
        let (mut x, mut y) = self.pt;
        let width = self.w as i32 - 1;
        let height = self.h as i32 - 1;
        let mut last_pos = self.lnbd;
        let mut prev = self.at(x - 1, y) as i32;
        while y < height {
            let mut p: i32 = 0;
            while x < width {
                // skip runs equal to `prev`, 8 bytes at a time (as OpenCV's SIMD scan)
                let row = y as usize * self.w;
                let pattern = u64::from_ne_bytes([prev as i8 as u8; 8]);
                while x + 8 <= width {
                    let i = row + x as usize;
                    let bytes: [i8; 8] = self.img[i..i + 8].try_into().unwrap();
                    let chunk = u64::from_ne_bytes(bytes.map(|b| b as u8));
                    if chunk != pattern {
                        break;
                    }
                    x += 8;
                }
                while x < width {
                    p = self.at(x, y) as i32;
                    if p != prev {
                        break;
                    }
                    x += 1;
                }
                if x >= width {
                    break;
                }
                if self.contour_scan(prev, p, &mut last_pos, x, y) {
                    self.lnbd = last_pos;
                    return true;
                }
                prev = p;
                if prev & MASK8_FLAGS != 0 {
                    last_pos.0 = x;
                }
                x += 1;
            }
            last_pos = (0, y + 1);
            x = 1;
            prev = 0;
            y += 1;
        }
        false
    }
}

/// cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
pub fn find_contours(mask: &[u8], h: usize, w: usize) -> Vec<Vec<[i32; 2]>> {
    find_contours_mode(mask, h, w, false)
}

/// cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
pub fn find_external_contours(mask: &[u8], h: usize, w: usize) -> Vec<Vec<[i32; 2]>> {
    find_contours_mode(mask, h, w, true)
}

fn find_contours_mode(mask: &[u8], h: usize, w: usize, external: bool) -> Vec<Vec<[i32; 2]>> {
    let (pw, ph) = (w + 2, h + 2);
    let mut img = vec![0i8; pw * ph];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] > 0 {
                img[(y + 1) * pw + x + 1] = 1;
            }
        }
    }
    let mut root = Node::new();
    root.is_hole = true;
    root.brect = (0, 0, pw as i32, ph as i32);
    let mut sc = Scanner { external, img, w: pw, h: ph, pt: (1, 1), lnbd: (0, 1), nbd: 2, tree: vec![root], ctable: [-1; 128] };
    while sc.find_next() {}
    // contourTreeToResults: depth first from the root, first child (the newest) first
    let mut out = Vec::with_capacity(sc.tree.len() - 1);
    let mut stack = vec![0usize];
    while let Some(i) = stack.pop() {
        let mut cur = sc.tree[i].first_child;
        if cur != -1 {
            while sc.tree[cur as usize].next != -1 {
                cur = sc.tree[cur as usize].next;
            }
            while cur != -1 {
                stack.push(cur as usize);
                cur = sc.tree[cur as usize].prev;
            }
        }
        if i != 0 {
            out.push(std::mem::take(&mut sc.tree[i].pts));
        }
    }
    out
}
