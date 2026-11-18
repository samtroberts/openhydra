//! Load-test measurement, run by hand:
//! `cargo test --release -p openhydra-network --test dup_cache_measure -- --ignored --nocapture`.
//! In its own test binary because it replaces the global allocator to count heap use.

/// Load-test measurement (run by hand, with `pex_cost_measure`): the heap the gossip duplicate
/// cache holds per remembered message id. Mirrors the patched library's `TimeCache` (a map from
/// id to expiry, and a queue of (id, expiry)) with ids as it builds them (sender peer id in base58
/// followed by the sequence number), counted by a counting allocator.
#[test]
#[ignore]
fn dup_cache_memory_measure() {
    use std::collections::{HashMap, VecDeque};
    struct ExpiringValue { _value: (), _expires: std::time::Instant }
    struct ExpiringElement { _element: Vec<u8>, _expires: std::time::Instant }
    for n in [30_000usize, 75_000, 300_000] {
        let ids: Vec<Vec<u8>> = (0..n)
            .map(|i| format!("{}{}", libp2p::PeerId::random().to_base58(), 1_700_000_000_000_000_000u64 + i as u64).into_bytes())
            .collect();
        let before = counting::live();
        let mut map: HashMap<Vec<u8>, ExpiringValue> = HashMap::new();
        let mut list: VecDeque<ExpiringElement> = VecDeque::new();
        let now = std::time::Instant::now();
        for id in &ids {
            map.insert(id.clone(), ExpiringValue { _value: (), _expires: now });
            list.push_back(ExpiringElement { _element: id.clone(), _expires: now });
        }
        let held = counting::live() - before;
        println!("DUP_CACHE n={n} id_len={} held_bytes={held} per_id={:.0}", ids[0].len(), held as f64 / n as f64);
        drop((map, list));
    }
}

mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicIsize, Ordering};
    struct Counting;
    static LIVE: AtomicIsize = AtomicIsize::new(0);
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            LIVE.fetch_add(l.size() as isize, Ordering::Relaxed);
            System.alloc(l)
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed);
            System.dealloc(p, l)
        }
    }
    #[global_allocator]
    static A: Counting = Counting;
    pub fn live() -> isize {
        LIVE.load(Ordering::Relaxed)
    }
}

