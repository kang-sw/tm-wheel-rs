use core::num::{NonZeroU32, NonZeroU64};

#[cfg(feature = "std")]
mod _std {}

/* -------------------------------------------- Trait ------------------------------------------- */

/// A backend allocator for [`TimerDriverBase`].
pub trait SlapAllocator<T> {
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

    /// # Safety
    ///
    /// The caller must ensure that all keys are valid and disjoint each other.
    unsafe fn get_many_mut_unchecked<const N: usize>(&mut self, keys: [u32; N]) -> [&mut T; N];
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
    A: SlapAllocator<TimerNode<T>>,
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
    A: SlapAllocator<TimerNode<T>> + Default,
{
    fn default() -> Self {
        Self::new(A::default())
    }
}

impl<T, A: SlapAllocator<TimerNode<T>>, const PAGES: usize, const PAGE_SIZE: usize>
    TimerDriverBase<T, A, PAGES, PAGE_SIZE>
{
    pub const PAGES: usize = PAGES;
    const BITS: usize = PAGE_SIZE.next_power_of_two();
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
        let offset = expires_at.saturating_sub(now);
        let msb_idx = 63_u32.saturating_sub(offset.leading_zeros());
        let page = msb_idx as usize / Self::BITS;
        let cursor = now >> Self::BITS;
        let cursor_offset = offset >> (page * Self::BITS);
        let slot_index = (cursor + cursor_offset) & Self::MASK;

        (page, slot_index as usize)
    }

    pub fn expiration_limit(&self) -> TimePoint {
        1 << (PAGES * Self::BITS)
    }

    ///
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

    /// Remove all timers.
    pub fn clear(&mut self) {
        self.slab.clear();
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

        // TODO: Detect all cursor advances
        let mut rehash_head = ELEM_NIL;

        // TODO: Rehash all upward timers to the downward slots

        // From highest changed page ...
        let bit_diff = self.now ^ now;
        let page_hi = 63_usize.saturating_sub(bit_diff.leading_zeros() as _) / Self::BITS;
        let time_diff = now - self.now;

        for page in (1..=page_hi).rev() {
            //              ^ Exclude first page.

            // Check if it's the first case ...
            let page_max = Self::MASK << (page * Self::BITS);
            let mut cursor = (self.now >> (page * Self::BITS)) & Self::MASK;

            // There might be an edge case where the amount by which the timer advances is
            // exceptionally large. in such a scenario, we cannot merely compare the
            // cursor position using the masked bit hash, as it could potentially undergo
            // a complete rotation. for instance, even if the cursor position remains the
            // same, all timers on the page might have expired.
            let dst_cursor = if time_diff > page_max {
                (cursor + Self::MASK) & Self::MASK
            } else {
                (now >> (page * Self::BITS)) & Self::MASK
            };

            // Until cursor reaches dst_cursor, rehash all nodes to the downward pages.
            while cursor != dst_cursor {
                // Push to rehash candidate.
                let slot = &mut self.slots[page][cursor as usize];

                if slot.tail == ELEM_NIL {
                    debug_assert_eq!(slot.head, ELEM_NIL);
                    continue;
                }

                let node = self.slab.get_mut(slot.tail).unwrap();

                // Just create incomplete link, as we're going to rehash them all
                // immediately.
                node.next = rehash_head;
                slot.head = ELEM_NIL;
                slot.tail = ELEM_NIL;

                cursor = (cursor + 1) & Self::MASK;
            }
        }

        // Rehash all timers
        while rehash_head != ELEM_NIL {
            let node_idx = rehash_head;
            let node = self.slab.get_mut(node_idx).unwrap();
            rehash_head = node.next;

            let expiration = node.expiration.max(now);
            let (page, slot_index) = Self::page_and_slot(self.now, expiration);

            let slot = &mut self.slots[page][slot_index];
            node.prev = ELEM_NIL;
            node.next = slot.head;

            Self::assign_node_to_slot_front(&mut self.slab, slot, node_idx)
        }

        // TODO: Collect all expired nodes and put it to returned iterator, which will
        // dispose all expired + un-drained nodes when dropped.
        let mut expired_head = ELEM_NIL; // TODO

        self.now = now;

        TimerDriverDrainIter {
            driver: self,
            head: expired_head,
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
    A: SlapAllocator<TimerNode<T>>,
    const PAGES: usize,
    const PAGE_SIZE: usize,
> {
    driver: &'a mut TimerDriverBase<T, A, PAGES, PAGE_SIZE>,
    head: u32,
}
