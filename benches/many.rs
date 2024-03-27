use criterion::{criterion_group, criterion_main, Bencher, Criterion};

pub fn insert_10k<const PAGE: usize, const PAGE_SIZE: usize>(c: &mut Bencher) {
    let mut tm = timey::TimerDriver::<u32, PAGE, PAGE_SIZE>::default();
    tm.reserve(10_000);
}
