use core::num::{NonZeroU32, NonZeroU64};

mod inner {
    type ElemIndex = u32;
    const ELEM_NIL: ElemIndex = u32::MAX;
    const PAGE_BITS: usize = 4;

    pub struct HashWheel<const PAGE: usize> {
        latest_advance: crate::TimePoint,
        slots: [[ElemIndex; 1 << PAGE_BITS]; PAGE],
    }

    impl<const PAGE: usize> Default for HashWheel<PAGE> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<const PAGE: usize> HashWheel<PAGE> {
        pub fn new() -> Self {
            Self {
                latest_advance: 0,
                slots: [[ELEM_NIL; 1 << PAGE_BITS]; PAGE],
            }
        }
    }
}

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
    fn remove(&mut self, key: u32);

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

/// Handle to a timer.
///
/// FIXME: Resolve timer Handle collision. Expired timer handle may affect alive one.
/// - Inserting generation
///     - 32 bit: still collision possibility
///     - 64 bit: memory overhead is significant
/// - Making timer handle 'owned'
///     - TimePoint + Handle -> For same slot,
// NOTE: The slab index(`.1`) is nonzero to make niche optimization available, therefore
// should be decremented by 1 before used as index.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimerHandle(NonZeroU32);

/// Basic timer driver with manually configurable page size / slab allocator.
#[derive(Default)]
pub struct TimerDriverBase<T, A: SlapAllocator<TimerNode<T>>, const PAGE: usize> {
    slab: A,
    wheel: inner::HashWheel<PAGE>,
    _phantom: std::marker::PhantomData<T>,
}

pub struct TimerNode<T> {
    value: T,
    expiration: TimePoint,
    next: u32,
    prev: u32,
}

impl<T, A: SlapAllocator<TimerNode<T>>, const PAGE: usize> TimerDriverBase<T, A, PAGE> {
    /// Insert new timer with expiration time point.
    pub fn insert(&mut self, value: T, expires_at: TimePoint) -> TimerHandle {
        todo!()
    }

    ///
    pub fn remove(&mut self, handle: TimerHandle) -> Option<T> {
        todo!()
    }

    /// Remove all timers.
    pub fn clear(&mut self) {
        self.slab.clear();
        self.wheel = Default::default();
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

/* ============================================================================================== */
/*                                            ITERATOR                                            */
/* ============================================================================================== */

pub struct TimerDriverDrainIter<'a, T, A: SlapAllocator<TimerNode<T>>, const PAGE: usize> {
    driver: &'a mut TimerDriverBase<T, A, PAGE>,
    head: u32,
}
