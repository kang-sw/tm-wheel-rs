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
const PAGE_BITS: usize = 6;
const PAGE_MASK: u64 = (1 << PAGE_BITS) - 1;

/// Basic timer driver with manually configurable page size / slab allocator.
pub struct TimerDriverBase<T, A: SlapAllocator<TimerNode<T>>, const PAGE: usize> {
    slab: A,
    id_alloc: u64,
    now: crate::TimePoint,
    slots: [[TimerPageSlot; 1 << PAGE_BITS]; PAGE],
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

impl<T, A, const PAGE: usize> Default for TimerDriverBase<T, A, PAGE>
where
    A: SlapAllocator<TimerNode<T>> + Default,
{
    fn default() -> Self {
        Self::new(A::default())
    }
}

impl<T, A: SlapAllocator<TimerNode<T>>, const PAGE: usize> TimerDriverBase<T, A, PAGE> {
    pub const PAGES: usize = PAGE;

    pub fn new(slab: A) -> Self {
        Self {
            slab,
            id_alloc: 0,
            now: 0,
            slots: [[Default::default(); 1 << PAGE_BITS]; PAGE],
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
        let (page, slot_index) = self.page_and_slot(expires_at);
        let slot = self.slots[page][slot_index];

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

        {
            let slot = &mut self.slots[page][slot_index];
            slot.head = new_node;

            if slot.tail == ELEM_NIL {
                slot.tail = new_node;
            }
        }

        if let Some(prev_node) = self.slab.get_mut(slot.head) {
            prev_node.prev = new_node;
        }

        TimerHandle(id, NonZeroU32::new(new_node + 1).unwrap())
    }

    fn page_and_slot(&self, expires_at: TimePoint) -> (usize, usize) {
        let offset = expires_at.saturating_sub(self.now);
        let msb_idx = 63_u32.saturating_sub(offset.leading_zeros());
        let page = msb_idx as usize / PAGE_BITS;
        let cursor = self.now >> PAGE_BITS;
        let cursor_offset = offset >> (page * PAGE_BITS);
        let slot_index = (cursor + cursor_offset) & PAGE_MASK;

        (page, slot_index as usize)
    }

    pub fn expiration_limit(&self) -> TimePoint {
        1 << (PAGE * PAGE_BITS)
    }

    ///
    pub fn remove(&mut self, handle: TimerHandle) -> Option<T> {
        let node = self.slab.get(handle.index())?;

        if node.id.get() != handle.id() {
            return None;
        }

        let prev = node.prev;
        let next = node.next;
        let node_expire_at = node.expiration;
        let (page, slot_index) = self.page_and_slot(node_expire_at);

        if let Some(prev_node) = self.slab.get_mut(prev) {
            prev_node.next = next;
        } else {
            debug_assert_eq!(self.slots[page][slot_index].head, handle.index());
            self.slots[page][slot_index].head = next;
        }

        if let Some(next_node) = self.slab.get_mut(next) {
            next_node.prev = prev;
        } else {
            debug_assert_eq!(self.slots[page][slot_index].tail, handle.index());
            self.slots[page][slot_index].tail = prev;
        }

        Some(self.slab.remove(handle.index()).value)
    }

    /// Remove all timers.
    pub fn clear(&mut self) {
        self.slab.clear();
    }

    /// Drain all timers. Prefer [`TimerDriverBase::clear`] if you don't need the drained
    /// values.
    pub fn drain(&mut self) -> TimerDriverDrainIter<T, A, PAGE> {
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
    pub fn advance(&mut self, now: TimePoint) -> TimerDriverDrainIter<T, A, PAGE> {
        assert!(now >= self.now, "time advanced backward!");

        // TODO: Detect all cursor advances
        let mut expired_head = ELEM_NIL;

        // TODO: Rehash all upward timers to the downward slots
        {
            // A linked list of rehashed nodes.
            let mut rehash_head = ELEM_NIL;
            let mut update_bits = self.now ^ now & !PAGE_MASK;
            //                       Exclude lowest page ^^^^^^^^^

            while update_bits != 0 {
                let msb_idx = 63_u32.saturating_sub(update_bits.leading_zeros());
                let page_idx = msb_idx as usize / PAGE_BITS;

                // Mark given page as handled.
                let bit_offset = page_idx * PAGE_BITS;
                update_bits &= !(PAGE_MASK << bit_offset);

                // Select all items to be rehashed.
                let page = &mut self.slots[page_idx];
                let mut cursor = (self.now >> bit_offset) & PAGE_MASK;
                let dst_cursor = (now >> bit_offset) & PAGE_MASK;

                while dst_cursor != cursor {
                    let slot = &mut page[cursor as usize];

                    if slot.tail != ELEM_NIL {
                        debug_assert_ne!(slot.head, ELEM_NIL);

                        // Append all nodes in slot to rehash list.
                        Self::link_back_mono(&mut self.slab, slot.tail, rehash_head);
                        rehash_head = slot.head;

                        // Clear slot
                        slot.head = ELEM_NIL;
                        slot.tail = ELEM_NIL;
                    }

                    // Rotate cursor within wrapped range.
                    cursor = (cursor + 1) & PAGE_MASK;
                }
            }

            // Rehash all
            while rehash_head != ELEM_NIL {
                rehash_head += 1;
                todo!()
            }
        }

        // TODO: At page 0, advance cursor and expire timers.

        // TODO: Collect all expired nodes and put it to returned iterator, which will
        // dispose all expired + un-drained nodes when dropped.

        // Update time point
        self.now = now;

        TimerDriverDrainIter {
            driver: self,
            head: expired_head,
        }
    }

    /// Create mono-directional link from `base` to `node`.
    fn link_back_mono(slab: &mut A, base: ElemIndex, node: ElemIndex) {
        todo!()
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

pub struct TimerDriverDrainIter<'a, T, A: SlapAllocator<TimerNode<T>>, const PAGE: usize> {
    driver: &'a mut TimerDriverBase<T, A, PAGE>,
    head: u32,
}
