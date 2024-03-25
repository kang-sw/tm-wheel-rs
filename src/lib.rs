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

        Self::link_node_to_slot(&mut self.slab, slot, new_node);
        TimerHandle(id, NonZeroU32::new(new_node + 1).unwrap())
    }

    fn link_node_to_slot(slab: &mut A, slot: &mut TimerPageSlot, node_idx: ElemIndex) {
        if slot.head == ELEM_NIL {
            slot.tail = node_idx;
        } else {
            let head = slot.head;
            let head_node = slab.get_mut(head).unwrap();
            head_node.prev = node_idx;
        }

        slot.head = node_idx;
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

        let node_expire_at = node.expiration;
        let (page, slot_index) = self.page_and_slot(node_expire_at);

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

        // From highest changed page ...
        let bit_diff = self.now ^ now;
        let page = 63_u64.saturating_sub(bit_diff.leading_zeros() as _) as usize / PAGE_BITS;
        let time_diff = now - self.now;

        for page in (0..=page).rev() {
            // check how many cursors need to be advanced. There's two cases:
            // - Timer advance was 'very large', which just rotated all page by once.
            // - Timer advance was just normal, as bit difference tracking is enough.

            // Check if it's the first case ...
            let page_max = PAGE_MASK << (page * PAGE_BITS);
            let cursor = (self.now >> (page * PAGE_BITS)) & PAGE_MASK;
            let dst_cursor = if time_diff > page_max {
                (cursor + PAGE_MASK) & PAGE_MASK
            } else {
                (now >> (page * PAGE_BITS)) & PAGE_MASK
            };

            // Until cursor reaches dst_cursor, rehash all nodes to the downward pages.
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
