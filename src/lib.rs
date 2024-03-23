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
const PAGE_BITS: usize = 4;
const PAGE_MASK: u64 = (1 << PAGE_BITS) - 1;

/// Basic timer driver with manually configurable page size / slab allocator.
pub struct TimerDriverBase<T, A: SlapAllocator<TimerNode<T>>, const PAGE: usize> {
    slab: A,
    id_alloc: u64,
    now: crate::TimePoint,
    slots: [[ElemIndex; 1 << PAGE_BITS]; PAGE],
    _phantom: std::marker::PhantomData<T>,
}

pub struct TimerNode<T> {
    value: T,
    id: NonZeroU64,
    expiration: TimePoint,
    next: u32,
    prev: u32,
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
            slots: [[ELEM_NIL; 1 << PAGE_BITS]; PAGE],
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
        let slot_head = self.slots[page][slot_index];

        let node = TimerNode {
            value,
            id: {
                self.id_alloc += 1;
                self.id_alloc += (self.id_alloc == 0) as u64;
                NonZeroU64::new(self.id_alloc).unwrap()
            },
            expiration: expires_at,
            next: slot_head,
            prev: ELEM_NIL,
        };

        let id = node.id.get();
        let new_node = self.slab.insert(node);
        self.slots[page][slot_index] = new_node;

        if let Some(prev_node) = self.slab.get_mut(slot_head) {
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

        if let Some(prev_node) = self.slab.get_mut(prev) {
            prev_node.next = next;
        } else {
            let (page, slot_index) = self.page_and_slot(node_expire_at);
            debug_assert_eq!(self.slots[page][slot_index], handle.index());

            self.slots[page][slot_index] = next;
        }

        if let Some(next_node) = self.slab.get_mut(next) {
            next_node.prev = prev;
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
    /// # Panics
    ///
    /// Panics if the given time point is less than the current time point. Same time
    /// point doesn't cause panic, as it's meaningful for timer insertions that are
    /// *already expired*.
    pub fn advance(&mut self, now: TimePoint) -> TimerDriverDrainIter<T, A, PAGE> {
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
