//! Ownership-only counter: claiming never changes an output's arithmetic.
use std::{
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering},
};

// Separate the counter and read-only metadata on both 64- and 128-byte lines.
#[repr(align(128))]
pub(super) struct Cursor(AtomicUsize);
impl Cursor {
    pub(super) fn new() -> Self {
        Self(AtomicUsize::new(0))
    }
    pub(super) fn claim(&self, units: usize, chunk: usize) -> Option<Range<usize>> {
        debug_assert!(chunk > 0);
        // Cap at units, avoiding wrap even after all ranks observe exhaustion.
        // Relaxed is sufficient: the host separately publishes and joins jobs.
        let mut start = self.0.load(Ordering::Relaxed);
        loop {
            if start >= units {
                return None;
            }
            let end = start + chunk.min(units - start);
            match self
                .0
                .compare_exchange_weak(start, end, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return Some(start..end),
                Err(observed) => start = observed,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn concurrent_claims_cover_each_unit_once_including_tails() {
        for units in [0, 1, 5, 257, 5120] {
            for chunk in [1, 7, 64, 8192] {
                let cursor = Cursor::new();
                let owners: Vec<_> = (0..units).map(|_| AtomicUsize::new(0)).collect();
                std::thread::scope(|scope| {
                    for _ in 0..4 {
                        let (cursor, owners) = (&cursor, &owners);
                        scope.spawn(move || {
                            while let Some(range) = cursor.claim(units, chunk) {
                                for i in range {
                                    owners[i].fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        });
                    }
                });
                assert!(owners.iter().all(|n| n.load(Ordering::Relaxed) == 1));
                assert!(cursor.claim(units, chunk).is_none());
            }
        }
    }
    #[test]
    fn exhaustion_and_last_chunk_cannot_wrap() {
        let cursor = Cursor::new();
        assert_eq!(cursor.claim(usize::MAX, usize::MAX), Some(0..usize::MAX));
        assert!(cursor.claim(usize::MAX, usize::MAX).is_none());
    }
}
