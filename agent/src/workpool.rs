// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! A minimal fixed-size worker pool for the provider serve loop.
//!
//! The serve loop ([`crate::provider::Provider::run_inbound`]) was strictly serial — it
//! polled one inbound request, ran the *entire* (blocking) inference, replied, then polled
//! the next. A single long generation (or, with M2.3 enforcement, a deliberate throttle
//! delay for a leecher) therefore head-of-line-blocked every other consumer. This pool
//! decouples polling from serving: the loop enqueues jobs and `n` workers run them
//! concurrently, so one slow request no longer stalls the rest.
//!
//! Bounded on purpose: the external engine has its own concurrency limit, and unbounded
//! thread-per-request would let a burst exhaust memory. `n` caps in-flight serves; excess
//! requests queue.
//!
//! Pure & swarm-free (jobs are opaque `FnOnce`s), so the concurrency mechanism is unit-
//! tested here without standing up a node.

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A unit of work handed to the pool. Boxed + `Send` so it can cross to a worker thread.
type Job = Box<dyn FnOnce() + Send + 'static>;

/// A fixed-size pool of worker threads draining a shared queue. Dropping the pool closes
/// the queue, lets workers finish everything already submitted, then joins them.
pub struct WorkerPool {
    /// `Option` so [`Drop`] can drop the sender *before* joining — closing the channel is
    /// what tells idle workers to exit once the queue drains.
    tx: Option<Sender<Job>>,
    handles: Vec<JoinHandle<()>>,
}

impl WorkerPool {
    /// Spawn a pool of `n` workers (clamped to ≥1). Each worker loops: take the next job
    /// from the shared queue and run it. Taking the job releases the queue lock *before*
    /// running it, so jobs execute concurrently — the lock only serialises the brief handoff.
    pub fn new(n: usize) -> Self {
        let n = n.max(1);
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let handles = (0..n)
            .map(|_| {
                let rx = Arc::clone(&rx);
                std::thread::spawn(move || loop {
                    // Lock only to receive; drop the guard before running the job so other
                    // workers can pull the next one meanwhile.
                    let job = {
                        let guard = match rx.lock() {
                            Ok(g) => g,
                            Err(_) => break, // a worker panicked holding the lock; bail
                        };
                        guard.recv()
                    };
                    match job {
                        Ok(job) => job(),
                        Err(_) => break, // sender dropped and queue drained → exit
                    }
                })
            })
            .collect();
        Self { tx: Some(tx), handles }
    }

    /// Enqueue `job` for the next free worker. A no-op if the pool is shutting down.
    pub fn submit<F: FnOnce() + Send + 'static>(&self, job: F) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Box::new(job)); // receiver only gone during shutdown
        }
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        // Close the queue: workers finish what's already submitted, then recv() returns Err.
        self.tx.take();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn runs_every_submitted_job() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let pool = WorkerPool::new(4);
            for _ in 0..100 {
                let c = Arc::clone(&counter);
                pool.submit(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                });
            }
            // Drop joins after the queue drains — every job has run by here.
        }
        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn jobs_run_concurrently_not_serially() {
        // Each job parks briefly; with a serial executor the peak in-flight count would be
        // 1. A pool of 4 must show real overlap, proving one slow job can't block the rest.
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        {
            let pool = WorkerPool::new(4);
            for _ in 0..8 {
                let in_flight = Arc::clone(&in_flight);
                let peak = Arc::clone(&peak);
                pool.submit(move || {
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(50));
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                });
            }
        }
        assert!(
            peak.load(Ordering::SeqCst) >= 2,
            "expected concurrent execution, peaked at {}",
            peak.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn zero_workers_is_clamped_to_one() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let pool = WorkerPool::new(0);
            let c = Arc::clone(&counter);
            pool.submit(move || {
                c.fetch_add(1, Ordering::SeqCst);
            });
        }
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
