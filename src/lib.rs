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

/* ============================================================================================== */
/*                                           TIMER BASE                                           */
/* ============================================================================================== */

/// A type alias for internal time point representation. It can be any unsigned interger
/// type, where the representation is arbitrary.
pub type TimePoint = u64;
type TimerGenId = u64;

type ElemIndex = u32;
const ELEM_NIL: ElemIndex = u32::MAX;

/// Basic timer driver with manually configurable page size / slab allocator.
pub struct TimerDriverBase<
    T,
    A: SlabAllocator<TimerNode<T>>,
    const PAGES: usize,
    const PAGE_SIZE: usize,
> {
    slab: A,
    id_alloc: u64,
    now: crate::TimePoint,
    slots: [[TimerPageSlot; PAGE_SIZE]; PAGES],
    _phantom: std::marker::PhantomData<T>,
}

pub struct TimerNode<T> {
    value: T,
    id: NonZeroU64,
    expiration: TimePoint,
    next: u32,
    prev: u32,
}

#[derive(Debug, Clone, Copy)]
struct TimerPageSlot {
    head: ElemIndex,
    tail: ElemIndex,
}

impl Default for TimerPageSlot {
    fn default() -> Self {
        Self {
            head: ELEM_NIL,
            tail: ELEM_NIL,
        }
    }
}

impl<T, A, const PAGES: usize, const PAGE_SIZE: usize> Default
    for TimerDriverBase<T, A, PAGES, PAGE_SIZE>
where
    A: SlabAllocator<TimerNode<T>> + Default,
{
    fn default() -> Self {
        Self::new(A::default())
    }
}

impl<T, A: SlabAllocator<TimerNode<T>>, const PAGES: usize, const PAGE_SIZE: usize>
    TimerDriverBase<T, A, PAGES, PAGE_SIZE>
{
    pub const PAGES: usize = PAGES;
    const BITS: usize = PAGE_SIZE.trailing_zeros() as usize;
    const MASK: u64 = PAGE_SIZE as u64 - 1;

    pub fn new(slab: A) -> Self {
        debug_assert!(PAGE_SIZE.is_power_of_two());

        Self {
            slab,
            id_alloc: 0,
            now: 0,
            slots: [[Default::default(); PAGE_SIZE]; PAGES],
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
        let (page, slot_index) = Self::page_and_slot(self.now, expires_at);
        let slot = &mut self.slots[page][slot_index];

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

        Self::assign_node_to_slot_front(&mut self.slab, slot, new_node);
        TimerHandle(id, NonZeroU32::new(new_node + 1).unwrap())
    }

    fn assign_node_to_slot_front(slab: &mut A, slot: &mut TimerPageSlot, node_idx: ElemIndex) {
        debug_assert!(slab.get(node_idx).unwrap().next == slot.head);

        if slot.head == ELEM_NIL {
            slot.tail = node_idx;
        } else {
            let head = slot.head;
            let head_node = slab.get_mut(head).unwrap();
            head_node.prev = node_idx;
        }

        slot.head = node_idx;
    }

    fn page_and_slot(now: TimePoint, expires_at: TimePoint) -> (usize, usize) {
        // FIXME: From page 1 to above, there's always unused slot present for index 0.
        // Perhaps we can fix this via extending page coverage, like in 4-bit setup
        // extending second page to cover 16~272(0~15) range, however, this might add
        // extra overhead during hashing, which is called inside hot path(rehashing).
        //
        // For now, we just let the memory space overhead present, which is usually small
        // for most of setups. (which significantly is reduced by increasing PAGE_SIZE)
        let offset = expires_at.saturating_sub(now);
        let msb_idx = 63_u32.saturating_sub(offset.leading_zeros());
        let page = msb_idx as usize / Self::BITS;
        let cursor = now >> Self::BITS;
        let cursor_offset = offset >> (page * Self::BITS);
        let slot_index = (cursor + cursor_offset) & Self::MASK;

        (page, slot_index as usize)
    }

    /// Get the maximum amount of time that can be set for a timer. Returned value is
    /// offset time from the current time point.
    pub fn expiration_limit(&self) -> u64 {
        1 << (PAGES * Self::BITS)
    }

    /// Remove given timer handle from the driver.
    pub fn remove(&mut self, handle: TimerHandle) -> Option<T> {
        let node = self.slab.get(handle.index())?;

        if node.id.get() != handle.id() {
            return None;
        }

        let node_expire_at = node.expiration;
        let (page, slot_index) = Self::page_and_slot(self.now, node_expire_at);

        let slot = &mut self.slots[page][slot_index];
        Self::unlink(&mut self.slab, slot, handle.index());

        Some(self.slab.remove(handle.index()).value)
    }

    fn unlink(slab: &mut A, slot: &mut TimerPageSlot, node_idx: ElemIndex) {
        let node = slab.get(node_idx).unwrap();
        let prev = node.prev;
        let next = node.next;

        if prev != ELEM_NIL {
            let prev_node = slab.get_mut(prev).unwrap();
            prev_node.next = next;
        } else {
            debug_assert_eq!(slot.head, node_idx);
            slot.head = next;
        }

        if next != ELEM_NIL {
            let next_node = slab.get_mut(next).unwrap();
            next_node.prev = prev;
        } else {
            debug_assert_eq!(slot.tail, node_idx);
            slot.tail = prev;
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

    /// Drain all timers. Prefer [`TimerDriverBase::clear`] if you don't need the drained
    /// values.
    pub fn drain(&mut self) -> TimerDriverDrainIter<T, A, PAGES, PAGE_SIZE> {
        todo!()
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
    pub fn advance(&mut self, now: TimePoint) -> TimerDriverDrainIter<T, A, PAGES, PAGE_SIZE> {
        assert!(now >= self.now, "time advanced backward!");

        // Detect all cursor advances
        let mut expired_head = ELEM_NIL;
        let mut expired_tail = ELEM_NIL;

        // From highest changed page ...
        let bit_diff = self.now ^ now;
        let page_hi = 63_usize.saturating_sub(bit_diff.leading_zeros() as _) / Self::BITS;
        let time_diff = now - self.now;

        for page in (0..=page_hi).rev() {
            // There might be an edge case where the amount by which the timer advances is
            // exceptionally large. in such a scenario, we cannot merely compare the
            // cursor position using the masked bit hash, as it could potentially undergo
            // a complete rotation. for instance, even if the cursor position remains the
            // same, all timers on the page might have expired.
            let page_max_time = Self::MASK << (page * Self::BITS);
            let mut cursor = (self.now >> (page * Self::BITS)) & Self::MASK;

            let dst_cursor = if time_diff > page_max_time {
                (cursor + Self::MASK) & Self::MASK
            } else {
                (now >> (page * Self::BITS)) & Self::MASK
            };

            while cursor != dst_cursor {
                let slot = &mut self.slots[page][cursor as usize];
                let mut slot_head = replace(&mut slot.head, ELEM_NIL);
                let slot_tail = replace(&mut slot.tail, ELEM_NIL);

                if page > 0 {
                    // Rehash every item within this slot.
                    while slot_head != slot_tail {
                        let node_idx = slot_head;
                        let node = self.slab.get_mut(node_idx).unwrap();
                        slot_head = node.next;

                        let expiration = node.expiration.max(now);
                        let (new_page, new_slot) = Self::page_and_slot(now, expiration);

                        let new_slot = &mut self.slots[new_page][new_slot];
                        node.next = new_slot.head;
                        Self::assign_node_to_slot_front(&mut self.slab, new_slot, node_idx);
                    }
                } else {
                    if slot_head == ELEM_NIL {
                        continue;
                    }

                    // append all expired pages to output expiration list
                    debug_assert!({ self.slab.get(slot_head).unwrap().expiration < now });

                    if let Some(expired) = self.slab.get_mut(expired_head) {
                        expired.prev = slot_tail;

                        let tail = self.slab.get_mut(slot_tail).unwrap();
                        tail.next = expired_head;
                    }

                    if expired_tail == ELEM_NIL {
                        expired_tail = slot_tail;
                    }

                    expired_head = slot_head;
                }

                cursor = (cursor + 1) & Self::MASK;
            }
        }

        // update timings
        self.now = now;

        TimerDriverDrainIter {
            driver: self,
            head: expired_head,
            tail: expired_tail,
        }
    }
}

/* ----------------------------------------- Handle Type ---------------------------------------- */

/// Handle to a timer.
///
/// FIXME: Resolve timer Handle collision. Expired timer handle may affect alive one.
/// - Inserting generation
///     - 32 bit: still collision possibility
///     - 64 bit: memory overhead is significant
/// - Making timer handle 'owned'
///     - TimePoint + Handle
///     - No reuse/collision since handle can't be copied
///     - No memory overhead
///     - If TimePoint is same: never collides with existing handle
///         - XXX: Maybe collide with expired handle??
///             - FIXME: Expired, but not `remove`d handle -> can invalidate valid timer.
///     - If TimePoint is different: it's just okay.
// NOTE: The slab index(`.1`) is nonzero to make niche optimization available, therefore
// should be decremented by 1 before used as index.
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
        for _ in self.by_ref() {}
    }
}

/* ============================================================================================== */
/*                                              TESTS                                             */
/* ============================================================================================== */
#[cfg(all(test, feature = "std"))]
mod tests {
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
            assert_eq!(tm.slots[page][slot].head, h.index());
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
    fn test_advance_expiration() {}
}
