//! Python's fractions.Fraction, for durations.

use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Frac {
    pub n: i64,
    pub d: i64,
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Frac {
    pub fn new(n: i64, d: i64) -> Frac {
        assert!(d != 0, "Fraction(n, 0)");
        let g = gcd(n, d).max(1);
        let s = if d < 0 { -1 } else { 1 };
        Frac { n: s * n / g, d: s * d / g }
    }
    pub fn int(n: i64) -> Frac {
        Frac { n, d: 1 }
    }
    pub const ZERO: Frac = Frac { n: 0, d: 1 };
    /// int(fraction): truncates toward zero
    pub fn trunc(self) -> i64 {
        self.n / self.d
    }
}

impl Add for Frac {
    type Output = Frac;
    fn add(self, o: Frac) -> Frac {
        Frac::new(self.n * o.d + o.n * self.d, self.d * o.d)
    }
}
impl Sub for Frac {
    type Output = Frac;
    fn sub(self, o: Frac) -> Frac {
        Frac::new(self.n * o.d - o.n * self.d, self.d * o.d)
    }
}
impl Mul for Frac {
    type Output = Frac;
    fn mul(self, o: Frac) -> Frac {
        Frac::new(self.n * o.n, self.d * o.d)
    }
}
impl Div for Frac {
    type Output = Frac;
    fn div(self, o: Frac) -> Frac {
        Frac::new(self.n * o.d, self.d * o.n)
    }
}
impl PartialOrd for Frac {
    fn partial_cmp(&self, o: &Frac) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Frac {
    fn cmp(&self, o: &Frac) -> Ordering {
        (self.n as i128 * o.d as i128).cmp(&(o.n as i128 * self.d as i128))
    }
}
