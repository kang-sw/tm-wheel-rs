#![doc=include_str!("../README.md")]

use core::{
    mem::replace,
    num::{NonZeroU32, NonZeroU64},
};

#[cfg(feature = "std")]
mod _std {
    impl<T> super::SlabAllocator<T> for slab::Slab<T> {
        fn insert(&mut self, value: T) -> u32 {
            self.insert(value) as _
        }

        fn remove(&mut self, key: u32) -> T {
            self.remove(key as _)
        }

        fn get(&self, key: u32) -> Option<&T> {
            self.get(key as _)
        }

        fn get_mut(&mut self, key: u32) -> Option<&mut T> {
            self.get_mut(key as _)
        }

        fn clear(&mut self) {
            self.clear();
        }

        fn len(&self) -> usize {
            self.len()
        }

        fn reserve(&mut self, additional: usize) {
            self.reserve(additional);
        }
    }

    pub type TimerDriver<T, const PAGES: usize, const PAGE_SIZE: usize> =
        super::TimerDriverBase<T, slab::Slab<super::TimerNode<T>>, PAGES, PAGE_SIZE>;
}

#[cfg(feature = "std")]
pub use _std::*;

/* -------------------------------------------- Trait ------------------------------------------- */

/// A backend allocator for [`TimerDriverBase`].
// TODO: document trait APIs for no-std implementors
pub trait SlabAllocator<T> {
    /// # Panics
    ///
    /// Panics if the key is not found
    fn insert(&mut self, value: T) -> u32;

    /// # Panics
    ///
    /// Panics if the key is not found
    fn remove(&mut self, key: u32) -> T;

    /// Get reference to the value by key.
    fn get(&self, key: u32) -> Option<&T>;

    /// Get mutable reference to the value by key.
    fn get_mut(&mut self, key: u32) -> Option<&mut T>;

    /// Clear all elements.
    fn clear(&mut self);

    /// Reserve extra space than `len()` for slab.
    ///
    /// For unsupported slab implementations(e.g. for embedded slabs), this method
    /// just does nothing.
    fn reserve(&mut self, additional: usize) {
        let _ = additional;
    }

    /// Get the number of elements.
    fn len(&self) -> usize;

    /// Check if slab allocator is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

trait SlabExt<T>
where
    Self: SlabAllocator<T>,
{
    fn at(&self, key: u32) -> &T {
        self.get(key).unwrap()
    }

    fn at_mut(&mut self, key: u32) -> &mut T {
        self.get_mut(key).unwrap()
    }
}

impl<T, A> SlabExt<T> for A where A: SlabAllocator<T> {}

/* ============================================================================================== */
/*                                           TIMER BASE                                           */
/* ============================================================================================== */

/// A type alias for internal time point representation. It can be any unsigned interger type,
/// where the representation is arbitrary.
pub type TimePoint = u64;

/// A type alias to represent delta time between two time points.
pub type TimeOffset = u64;

type TimerGenId = u64;
type ElemIndex = u32;

const ELEM_NIL: ElemIndex = u32::MAX;

/// Basic timer driver with manually configurable page size / slab allocator.
///
/// - `WHEELS`: Number of timing wheel hierarchy.
/// - `WHEEL_LENGTH`: Number of slots in each page. This MUST be power of 2.
#[derive(Debug, Clone)]
pub struct TimerDriverBase<
    T,
    A: SlabAllocator<TimerNode<T>>,
    const LEVELS: usize,
    const WHEEL_SIZE: usize,
> {
    slab: A,
    id_alloc: u64,
    now: crate::TimePoint,
    wheels: [TimerWheel<WHEEL_SIZE>; LEVELS],
    _phantom: std::marker::PhantomData<T>,
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TimerNode<T> {
    value: T,
    id: NonZeroU64,
    expiration: TimePoint,
    next: u32,
    prev: u32,
}

#[derive(Debug, Clone, Copy)]
struct TimerWheelBucket {
    head: ElemIndex,
    tail: ElemIndex,
}

impl Default for TimerWheelBucket {
    fn default() -> Self {
        Self {
            head: ELEM_NIL,
            tail: ELEM_NIL,
        }
    }
}

impl<T, A, const LEVELS: usize, const WHEEL_SIZE: usize> Default
    for TimerDriverBase<T, A, LEVELS, WHEEL_SIZE>
where
    A: SlabAllocator<TimerNode<T>> + Default,
{
    fn default() -> Self {
        Self::new(A::default())
    }
}

impl<T, A: SlabAllocator<TimerNode<T>>, const LEVELS: usize, const WHEEL_SIZE: usize>
    TimerDriverBase<T, A, LEVELS, WHEEL_SIZE>
{
    pub const LEVELS: usize = LEVELS;
    const BITS: usize = WHEEL_SIZE.trailing_zeros() as usize;
    const MASK: u64 = WHEEL_SIZE as u64 - 1;

    const E_BITS: usize = Self::BITS << 1;
    const E_MASK: u64 = (WHEEL_SIZE << 1) as u64 - 1;

    pub fn new(slab: A) -> Self {
        debug_assert!(WHEEL_SIZE.is_power_of_two());

        Self {
            slab,
            id_alloc: 0,
            now: 0,
            wheels: [Default::default(); LEVELS],
            _phantom: Default::default(),
        }
    }

    /// Insert new timer with expiration time point.
    ///
    /// # Panics
    ///
    /// - Specified timer expiration is out of range.
    /// - Active timer instance is larger than 2^32-1
    pub fn insert(&mut self, value: T, expires_at: TimePoint) -> TimerHandle {
        let (page, slot_index) = Self::level_and_bucket(self.now, expires_at);
        let slot = &mut self.wheels[page][slot_index];

        let node = TimerNode {
            value,
            id: {
                self.id_alloc += 1;
                self.id_alloc += (self.id_alloc == 0) as u64;
                NonZeroU64::new(self.id_alloc).unwrap()
            },
            expiration: expires_at,
            next: slot.head,
            prev: ELEM_NIL,
        };

        let id = node.id.get();
        let new_node = self.slab.insert(node);

        Self::assign_node_to_bucket_front(&mut self.slab, slot, new_node);
        TimerHandle(id, NonZeroU32::new(new_node + 1).unwrap())
    }

    fn assign_node_to_bucket_front(
        slab: &mut A,
        bucket: &mut TimerWheelBucket,
        node_idx: ElemIndex,
    ) {
        debug_assert!(slab.at(node_idx).next == bucket.head);

        if bucket.head == ELEM_NIL {
            bucket.tail = node_idx;
        } else {
            let head = bucket.head;
            let head_node = slab.at_mut(head);
            head_node.prev = node_idx;
        }

        bucket.head = node_idx;
    }

    fn level_and_bucket(now: TimePoint, expires_at: TimePoint) -> (usize, usize) {
        // TODO: Re-architect this.
        // - Once registered, offset time range which cannot be fit to single BIT_MASK range should
        //   be tossed to upstream level.
        // - Let's assume the PAGE_BITS is 4. then the timer item `A` with expiration 1_0000 will
        //   be registered to next level.
        // - The original logic tries to detect bit flip of current level. In this case, the
        //   current time 0000~1111 won't touch the second level, where the item `A` already has
        //   the timeout which is less than 1_0000.
        // - Therefore, every `ONE` of each level should be treated specially, since they easily be
        //   invalidated as the timestamp advances.
        // - Should we re-hash every first bucket of each level on every time step advance? it's
        //   not realistic and very inefficient.
        // - The point is, making each level *intersect* and letting the lowest bucket to be
        //   distributed into lower buckets, and storing items with bucket offset larger than 1 to
        //   be held on current level.

        let timeout = expires_at - now;
        let msb_pos = (63u32 - 1).saturating_sub(timeout.leading_zeros());
        //                        ^^^ To downgrade the last bucket of upper level ...
        // e.g. for 4bit bucket, 1_xxxx will be treated level 0. 1x_xxxx is level 1.
        // -                     ^ msb_pos=4-1=3:=lv.0           ^ msb_pos=5-1=4:=lv.1

        let level = msb_pos as usize / Self::BITS;
        let cursor = now >> (level * Self::BITS);
        let cursor_offset = timeout >> (level * Self::BITS);

        // We use extended mask here, to intrude upper level's LSB component.
        let bucket = (cursor + cursor_offset) & Self::E_MASK;

        (level, bucket as usize)
    }

    /// Unit of time per slot in given page.
    fn time_units(level: usize) -> TimeOffset {
        1 << (level * Self::BITS)
    }

    /// Get the maximum amount of time that can be set for a timer. Returned value is
    /// offset time from the current time point.
    pub fn expiration_limit(&self) -> TimeOffset {
        1 << (LEVELS * Self::BITS)
    }

    /// Remove given timer handle from the driver.
    pub fn remove(&mut self, handle: TimerHandle) -> Option<T> {
        let node = self.slab.get(handle.index())?;

        if node.id.get() != handle.id() {
            return None;
        }

        let node_expire_at = node.expiration;
        let (level, bucket_idx) = Self::level_and_bucket(self.now, node_expire_at);

        let slot = &mut self.wheels[level][bucket_idx];
        Self::unlink(&mut self.slab, slot, handle.index());

        Some(self.slab.remove(handle.index()).value)
    }

    fn unlink(slab: &mut A, bucket: &mut TimerWheelBucket, node_idx: ElemIndex) {
        let node = slab.at(node_idx);
        let prev = node.prev;
        let next = node.next;

        if prev != ELEM_NIL {
            let prev_node = slab.at_mut(prev);
            prev_node.next = next;
        } else {
            debug_assert_eq!(bucket.head, node_idx);
            bucket.head = next;
        }

        if next != ELEM_NIL {
            let next_node = slab.at_mut(next);
            next_node.prev = prev;
        } else {
            debug_assert_eq!(bucket.tail, node_idx);
            bucket.tail = prev;
        }
    }

    /// Get the number of registered timers.
    pub fn len(&self) -> usize {
        self.slab.len()
    }

    /// Check if the driver is empty.
    pub fn is_empty(&self) -> bool {
        self.slab.is_empty()
    }

    /// Get the current time point.
    pub fn now(&self) -> TimePoint {
        self.now
    }

    /// Reserve the space for additional timers. If underlying [`SlabAllocator`] doesn't
    /// support it, this method does nothing.
    pub fn reserve(&mut self, additional: usize) {
        self.slab.reserve(additional);
    }

    /// Remove all timers. Time point regression is only allowed if the driver is empty.
    pub fn reset(&mut self, now: TimePoint) {
        self.slab.clear();
        self.now = now;
    }

    /// Suggested timing for next wakeup. It is guaranteed that none of the timers will be
    /// overslept until returned time point.
    ///
    /// If there's no nearest timer in lowest level page, it'll return the nearest rehash
    /// timing which will drag down the timers from higher pages, resulting no actual
    /// timer expiration output.
    pub fn nearest_wakeup(&mut self) -> TimeOffset {
        if self.is_empty() {
            // No timers are registered ... don't even need to wakeup.
            return TimeOffset::MAX;
        }

        // Iterate all pages from the lowest slot, until find non-empty one.
        for wheel_idx in 0..Self::LEVELS {
            let cursor_base = (self.now >> (wheel_idx * Self::BITS)) & Self::MASK;
            let wheel = &self.wheels[wheel_idx];

            for bucket_offset in 0..2 * WHEEL_SIZE {
                let cursor = (cursor_base + bucket_offset as u64) & Self::MASK;
                let bucket = &wheel[cursor as usize];

                if bucket.head != ELEM_NIL {
                    return self.now + Self::time_units(wheel_idx);
                }
            }
        }

        // This is just a logic error
        unreachable!("no timers are registered, but is_empty() returned false")
    }

    /// Advance timer driver to the given time point.
    ///
    /// # Warning
    ///
    /// Expiration order is not guaranteed. It's caller's responsibility to handle
    /// correct expiration order.
    ///
    /// # Panics
    ///
    /// Panics if the given time point is less than the current time point. Same time
    /// point doesn't cause panic, as it's meaningful for timer insertions that are
    /// *already expired*.
    pub fn advance_to(
        &mut self,
        time_point: TimePoint,
    ) -> TimerDriverDrainIter<T, A, LEVELS, WHEEL_SIZE> {
        self.advance(time_point - self.now)
    }

    /// Advance timer by timer amount.
    pub fn advance(
        &mut self,
        advance: TimeOffset,
    ) -> TimerDriverDrainIter<T, A, LEVELS, WHEEL_SIZE> {
        // Detect all cursor advances
        let mut expired_head = ELEM_NIL;
        let mut expired_tail = ELEM_NIL;

        // Firstly expire all lowest levels
        let next_tp = self.now + advance;
        let advance_amount = advance - self.now;

        {
            let mut cursor = self.now & Self::E_MASK;
            let dst_cursor = if advance_amount > Self::E_MASK {
                Self::E_MASK
            } else {
                next_tp & Self::E_MASK
            };

            let wheel = &mut self.wheels[0];
            loop {
                // Always clear out current cursor position
                let bucket = &mut wheel[cursor as usize];
                let head = replace(&mut bucket.head, ELEM_NIL);
                let tail = replace(&mut bucket.tail, ELEM_NIL);

                // For lowest page, append is always O(1) for any number of nodes
                if expired_head == ELEM_NIL {
                    expired_head = head;
                    expired_tail = tail;
                } else {
                    self.slab.at_mut(expired_tail).next = head;
                    self.slab.at_mut(head).prev = expired_tail;

                    expired_tail = tail
                }

                if cursor == dst_cursor {
                    break;
                }

                cursor = (cursor + 1) & Self::E_MASK
            }
        }

        // From highest changed page ...
        let bit_diff = self.now ^ advance;

        // Unlike the logic inside `level_and_bucket`, this method account for the bit flip of the
        // lowest position.
        let mut page_hi = 63_usize.saturating_sub(bit_diff.leading_zeros() as _) / Self::BITS;
        page_hi = page_hi.min(LEVELS - 1);

        // List of all rehash targets
        let mut rehash_head = ELEM_NIL;

        // Logic
        // - For any wheel greater than level 0, rehash all items until next cursor + 1.
        // - See `level_and_bucket`
        for level in 1..=page_hi {
            // TODO: Reimplement re-hashing logic

            let wheel = &mut self.wheels[level];
            let level_bits = level * Self::BITS;
            let mut cursor = (self.now >> level_bits) & Self::E_MASK;

            // Based on above logic, current and the next cursor position MUST be empty, as the
            // hashing method never target those two.
            debug_assert_eq!(wheel[cursor as usize].head, ELEM_NIL);
            debug_assert_eq!(wheel[((cursor + 1) & Self::E_MASK) as usize].head, ELEM_NIL);

            // next cursor and next cursor + 1 are rehashed.
            let next_cursor = next_tp >> level_bits;
            let cursor_dst = (next_cursor + 1) & Self::E_MASK;
            let mut bucket_expiration = (self.now >> level_bits) << level_bits;

            loop {
                bucket_expiration += Self::time_units(level);
                let bucket = &mut wheel[cursor as usize];

                let bucket_head = replace(&mut bucket.head, ELEM_NIL);
                let bucket_tail = replace(&mut bucket.tail, ELEM_NIL);

                if bucket_expiration >= next_tp {
                    // Append all items in this bucket to the expiration list
                    if expired_head == ELEM_NIL {
                        expired_head = bucket_head;
                        expired_tail = bucket_tail;
                    } else {
                        self.slab.at_mut(expired_tail).next = bucket_head;
                        self.slab.at_mut(bucket_head).prev = expired_tail;

                        expired_tail = bucket_tail;
                    }
                } else {
                    // Rehash all items within this bucket.
                    if rehash_head == ELEM_NIL {
                        rehash_head = bucket_head;
                    } else {
                        self.slab.at_mut(bucket_tail).next = rehash_head;
                        self.slab.at_mut(rehash_head).prev = bucket_tail;
                        rehash_head = bucket_head;
                    }
                }

                if cursor == cursor_dst {
                    break;
                }

                cursor += 1;
            }
        }

        // Flush pending rehash target nodes
        while rehash_head != ELEM_NIL {
            let node_idx = rehash_head;
            let node = self.slab.at_mut(rehash_head);
            rehash_head = node.next;

            let (level, bucket) = Self::level_and_bucket(self.now, node.expiration);
            let bucket = &mut self.wheels[level][bucket];
            node.next = bucket.head;
            node.prev = ELEM_NIL;

            Self::assign_node_to_bucket_front(&mut self.slab, bucket, node_idx);
        }

        // update timings
        self.now = advance;

        TimerDriverDrainIter {
            driver: self,
            head: expired_head,
            tail: expired_tail,
        }
    }
}

/* -------------------------------------- Timer Bucket Type ------------------------------------- */

#[derive(Debug, Clone, Copy)]
struct TimerWheel<const WHEEL_SIZE: usize> {
    buckets: [[TimerWheelBucket; WHEEL_SIZE]; 2],
}

impl<const WHEEL_SIZE: usize> TimerWheel<WHEEL_SIZE> {
    const MASK: u64 = WHEEL_SIZE as u64 - 1;
}

impl<const WHEEL_SIZE: usize> Default for TimerWheel<WHEEL_SIZE> {
    fn default() -> Self {
        Self {
            buckets: [[Default::default(); WHEEL_SIZE]; 2],
        }
    }
}

impl<const WHEEL_SIZE: usize> std::ops::Index<usize> for TimerWheel<WHEEL_SIZE> {
    type Output = TimerWheelBucket;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buckets[(index > WHEEL_SIZE) as usize][index & Self::MASK as usize]
    }
}

impl<const WHEEL_SIZE: usize> std::ops::IndexMut<usize> for TimerWheel<WHEEL_SIZE> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.buckets[(index > WHEEL_SIZE) as usize][index & Self::MASK as usize]
    }
}

/* ----------------------------------------- Handle Type ---------------------------------------- */

/// Handle to a created timer.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimerHandle(TimerGenId, NonZeroU32);

impl TimerHandle {
    fn id(&self) -> TimerGenId {
        self.0
    }

    fn index(&self) -> u32 {
        self.1.get() - 1
    }
}

/* ============================================================================================== */
/*                                            ITERATOR                                            */
/* ============================================================================================== */

pub struct TimerDriverDrainIter<
    'a,
    T,
    A: SlabAllocator<TimerNode<T>>,
    const PAGES: usize,
    const PAGE_SIZE: usize,
> {
    driver: &'a mut TimerDriverBase<T, A, PAGES, PAGE_SIZE>,
    head: ElemIndex,
    tail: ElemIndex,
}

impl<'a, T, A: SlabAllocator<TimerNode<T>>, const PAGES: usize, const PAGE_SIZE: usize> Iterator
    for TimerDriverDrainIter<'a, T, A, PAGES, PAGE_SIZE>
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.head == ELEM_NIL {
            return None;
        }

        let node_idx = self.head;
        let node = self.driver.slab.remove(node_idx);

        if self.head == self.tail {
            // This is required since we don't manipulate linked node's pointer.
            self.head = ELEM_NIL;
            self.tail = ELEM_NIL;
        } else {
            self.head = node.next;
        }

        Some(node.value)
    }
}

impl<'a, T, A: SlabAllocator<TimerNode<T>>, const PAGES: usize, const PAGE_SIZE: usize>
    DoubleEndedIterator for TimerDriverDrainIter<'a, T, A, PAGES, PAGE_SIZE>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.tail == ELEM_NIL {
            return None;
        }

        let node_idx = self.tail;
        let node = self.driver.slab.remove(node_idx);

        if self.head == self.tail {
            // same as above
            self.head = ELEM_NIL;
            self.tail = ELEM_NIL;
        } else {
            self.tail = node.prev;
        }

        Some(node.value)
    }
}

impl<'a, T, A: SlabAllocator<TimerNode<T>>, const PAGES: usize, const PAGE_SIZE: usize> Drop
    for TimerDriverDrainIter<'a, T, A, PAGES, PAGE_SIZE>
{
    fn drop(&mut self) {
        // consume all remaining elements
        for _ in self.by_ref() {}
    }
}

/* ============================================================================================== */
/*                                              TESTS                                             */
/* ============================================================================================== */
#[cfg(all(test, feature = "std"))]
mod tests {
    use std::collections::HashMap;

    use crate::TimePoint;

    #[test]
    fn test_insertion_and_structure_validity() {
        let mut tm = crate::TimerDriver::<u32, 4, 16>::default();
        let mut idx = {
            let mut gen = 0;
            move || {
                gen += 1;
                gen
            }
        };

        // Should be hashed to page 0, slot 10
        let mut insert_and_compare = |expire: TimePoint, page: usize, slot: usize| {
            let h = tm.insert(idx(), expire);
            assert_eq!(tm.wheels[page][slot].head, h.index());
        };

        insert_and_compare(10, 0, 10);
        insert_and_compare(20, 1, 1);
        insert_and_compare(30, 1, 1);
        insert_and_compare(40, 1, 2);
        insert_and_compare(50, 1, 3);
        insert_and_compare(60, 1, 3);
        insert_and_compare(60, 1, 3);
        insert_and_compare(255, 1, 15);

        insert_and_compare(257, 2, 1);
        insert_and_compare(258, 2, 1);
        insert_and_compare(512, 2, 2);
        insert_and_compare(768, 2, 3);

        assert_eq!(tm.len(), 12);
    }

    #[test]
    fn test_advance_expiration() {
        let mut tm = crate::TimerDriver::<u32, 4, 16>::default();

        /* ----------------------------------- Test First Page ---------------------------------- */
        tm.reset(1000);
        tm.insert(1, 1010);
        tm.insert(2, 1020);
        tm.insert(3, 1030);

        assert_eq!(tm.nearest_wakeup(), 1010);
        assert_eq!(tm.advance_to(1010).next(), Some(1));
        assert_eq!(tm.nearest_wakeup(), 1020);
        assert_eq!(tm.advance_to(1020).next(), Some(2));
        assert_eq!(tm.nearest_wakeup(), 1030);
        assert_eq!(tm.advance_to(1030).next(), Some(3));

        assert_eq!(tm.len(), 0);

        /* ------------------------------ Test Many Page & Removal ------------------------------ */
        const OFST: u64 = 102;
        tm.reset(OFST);
        tm.insert(1, OFST + 49133);
        tm.insert(2, OFST + 310);
        let h = tm.insert(3, OFST + 59166);
        tm.insert(4, OFST + 1411);

        assert_eq!(tm.len(), 4);
        assert_eq!(tm.remove(h), Some(3));
        assert_eq!(tm.len(), 3);

        dbg!(tm.nearest_wakeup());
        assert_eq!(tm.advance_to(OFST + 310).next(), Some(2));
        dbg!(tm.nearest_wakeup());
        assert_eq!(tm.advance_to(OFST + 999).next(), None);
        dbg!(tm.nearest_wakeup());
        assert_eq!(tm.advance_to(OFST + 1411).next(), Some(4));
        dbg!(tm.nearest_wakeup());
        assert_eq!(tm.advance_to(OFST + 49133).next(), Some(1));
        dbg!(tm.nearest_wakeup());
        assert_eq!(tm.advance_to(OFST + 60000).next(), None);

        assert_eq!(tm.len(), 0);

        /* ------------------------------------ Test Iterator ----------------------------------- */
        tm.reset(OFST);
        tm.insert(1, OFST + 49133);
        tm.insert(2, OFST + 310);
        tm.insert(3, OFST + 59166);
        tm.insert(4, OFST + 1411);

        assert_eq!(tm.len(), 4);
        assert_eq!(
            tm.advance_to(OFST + 60_000).collect::<Vec<_>>(),
            [2, 4, 1, 3]
        );
        assert_eq!(tm.len(), 0);
    }

    // todo: scaled test from generated inputs
    #[test]
    fn test_many_cases() {
        let mut rng = fastrand::Rng::with_seed(0);
        let mut active_handles = HashMap::new();
        let mut timer_id = 0;
        let num_iter = 100000;
        let mut now = 0u64;

        // frequent enough to see page shift
        let mut tm = crate::TimerDriver::<u32, 12, 4>::default();
        dbg!(tm.expiration_limit());

        for _ in 0..num_iter {
            now += rng.u64(1..128);

            for expired in tm.advance_to(now) {
                assert!(active_handles.remove(&expired).is_some_and(|v| v <= now));
            }

            for _ in 0..rng.usize(0..5) {
                let id = timer_id;
                timer_id += 1;

                let expire_at = now + rng.u64(1..10000);
                active_handles.insert(id, expire_at);
                tm.insert(id, expire_at);
            }
        }

        now = u64::MAX;

        for expired in tm.advance_to(u64::MAX) {
            assert!(active_handles.remove(&expired).is_some_and(|v| v <= now));
        }

        assert!(active_handles.is_empty());
    }
}
