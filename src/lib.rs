#![deny(unused, missing_docs)]
//! `MergeIter` is an iterator that returns the elements of two iterators in order, given both
//! iterators are sorted.
//!
//! **Important note**: This iterator only works as intended, if both input iterators are sorted.
//! There are no checks in place to validate this property.

#[cfg(test)]
#[macro_use]
extern crate proptest;

use std::{cmp::Reverse, collections::BinaryHeap, iter::Peekable};

/// A sorted iterator over two independent sorted iterators.
pub struct MergeIter<L, R, T>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    left: Peekable<L>,
    right: Peekable<R>,
    cmp_function: fn(&T, &T) -> bool,
}

impl<L, R, T> From<(L, R)> for MergeIter<L, R, T>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
    T: Ord,
{
    #[inline]
    fn from((left, right): (L, R)) -> Self {
        Self::new(left, right)
    }
}

impl<L, R, T> MergeIter<L, R, T>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
    T: Ord,
{
    /// Creates a new `MergeIter` that returns elements from both supplied iterators in order,
    /// given they are sorted.
    ///
    /// # Examples
    /// ```
    /// use merging_iterator::MergeIter;
    /// let a = vec![0, 2, 4, 6, 8];
    /// let b = vec![1, 3, 5, 7, 9];
    /// let merger = MergeIter::new(a, b);
    /// assert_eq!(
    ///     vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ///     merger.collect::<Vec<_>>()
    /// );
    /// ```
    #[inline]
    pub fn new<IL, IR>(left: IL, right: IR) -> Self
    where
        IL: IntoIterator<IntoIter = L, Item = T>,
        IR: IntoIterator<IntoIter = R, Item = T>,
    {
        Self {
            left: left.into_iter().peekable(),
            right: right.into_iter().peekable(),
            cmp_function: |a, b| a < b,
        }
    }
}

impl<L, R, T> MergeIter<L, R, T>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    /// Creates a new `MergeIter` that uses a custom ordering function.
    ///
    /// # Examples
    /// ```
    /// use merging_iterator::MergeIter;
    /// let a = vec![8, 6, 4, 2, 0];
    /// let b = vec![9, 7, 5, 3, 1];
    /// let cmp = |a: &u8, b: &u8| b < a;
    /// let merger = MergeIter::with_custom_ordering(a, b, cmp);
    /// assert_eq!(
    ///     vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ///     merger.collect::<Vec<_>>()
    /// );
    /// ```
    #[inline]
    pub fn with_custom_ordering<IL, IR>(left: IL, right: IR, cmp: fn(&T, &T) -> bool) -> Self
    where
        IL: IntoIterator<IntoIter = L, Item = T>,
        IR: IntoIterator<IntoIter = R, Item = T>,
    {
        Self {
            left: left.into_iter().peekable(),
            right: right.into_iter().peekable(),
            cmp_function: cmp,
        }
    }
}

impl<L, R, T> Iterator for MergeIter<L, R, T>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // Temporary enum to prevent issues with the borrow checker
        enum Next {
            Left,
            Right,
            None,
        }
        let n = match (self.left.peek(), self.right.peek()) {
            (Some(ref l), Some(ref r)) => {
                if (self.cmp_function)(l, r) {
                    Next::Left
                } else {
                    Next::Right
                }
            }
            (Some(_), None) => Next::Left,
            (None, Some(_)) => Next::Right,
            (None, None) => Next::None,
        };
        match n {
            Next::Left => self.left.next(),
            Next::Right => self.right.next(),
            Next::None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (l, lo) = self.left.size_hint();
        let (r, ro) = self.right.size_hint();
        (
            l + r,
            match (lo, ro) {
                (Some(lo), Some(ro)) => Some(lo + ro),
                // no predictable upper bound
                _ => None,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    proptest! {
        #[test]
        fn test_is_sorted_property(mut a: Vec<i32>, mut b: Vec<i32>) {
            a.sort_unstable();
            b.sort_unstable();
            let merger = MergeIter::new(a, b);
            assert!(is_sorted(merger));
        }

        #[test]
        fn test_is_sorted_property_for_multiple_iterators(
            mut a: Vec<i32>,
            mut b: Vec<i32>,
            mut c: Vec<i32>
        ) {
            a.sort_unstable();
            b.sort_unstable();
            c.sort_unstable();
            let merger = MergeIter::new(
                MergeIter::new(a, b),
                c
            );
            assert!(is_sorted(merger));
        }
    }

    #[test]
    fn test_merge_sorted_iterators() {
        let expected = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        let a = vec![1, 3, 5, 7, 9];
        let b = vec![2, 4, 6, 8];
        let merger = MergeIter::new(a, b);
        assert_eq!(expected, merger.collect::<Vec<_>>());

        let a = vec![1, 2, 3, 4, 5];
        let b = vec![6, 7, 8, 9];
        let merger = MergeIter::new(a, b);
        assert_eq!(expected, merger.collect::<Vec<_>>());

        let a = vec![3, 5, 6, 8];
        let b = vec![1, 2, 4, 7, 9];
        let merger = MergeIter::new(a, b);
        assert_eq!(expected, merger.collect::<Vec<_>>());

        let a = vec![];
        let b = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let merger = MergeIter::new(a, b);
        assert_eq!(expected, merger.collect::<Vec<_>>());

        let a = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let b = vec![];
        let merger = MergeIter::new(a, b);
        assert_eq!(expected, merger.collect::<Vec<_>>());
    }

    #[test]
    fn test_multiple_iterators() {
        let expected = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        let a = vec![1, 4, 7];
        let b = vec![2, 5, 8];
        let c = vec![3, 6, 9];
        let merger = MergeIter::new(a, b);
        let merger = MergeIter::new(c, merger);
        let merger = merger.collect::<Vec<_>>();
        assert_eq!(expected, merger);
        assert!(is_sorted(merger.iter()));
    }

    #[test]
    fn test_merge_unord() {
        struct UnOrd;

        let a = vec![UnOrd];
        let b = vec![UnOrd];
        let merger = MergeIter::with_custom_ordering(a, b, |_, _| true);
        let _ = merger.collect::<Vec<_>>();
    }

    fn is_sorted<I, T>(iter: I) -> bool
    where
        I: Iterator<Item = T>,
        T: Ord,
    {
        iter.fold((true, None), |(res, last), next| {
            (res && last.map(|v| v < next).unwrap_or(true), Some(next))
        })
        .0
    }
}

struct Takeble<T, I: Iterator<Item = T>> {
    iter: I,
    next: T,
}

impl<T, I: Iterator<Item = T>> Takeble<T, I> {
    pub fn try_new(mut iter: I) -> Option<Self> {
        let next = iter.next();
        next.map(|next| Self { iter, next })
    }

    pub fn take(self) -> (T, Option<Self>) {
        let Self { next, iter } = self;

        (next, Self::try_new(iter))
    }
}
impl<T: Ord, I: Iterator<Item = T>> PartialEq for Takeble<T, I> {
    fn eq(&self, other: &Self) -> bool {
        self.next == other.next
    }
}
impl<T: Ord, I: Iterator<Item = T>> Eq for Takeble<T, I> {}

impl<T: Ord, I: Iterator<Item = T>> PartialOrd for Takeble<T, I> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: Ord, I: Iterator<Item = T>> Ord for Takeble<T, I> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.next.cmp(&other.next)
    }
}

/// An iterator that merges sorted iterators into one sorted iterator.
pub struct MergeSortedIter<T: Ord, I: Iterator<Item = T>> {
    heap: BinaryHeap<Reverse<Takeble<T, I>>>,
}

impl<T: Ord, I: Iterator<Item = T>> MergeSortedIter<T, I> {
    /// Creates a new `MergeSortedIter` by merging the provided sorted iterators.
    pub fn new(sorted_iterators: impl Iterator<Item = I>) -> Self {
        Self {
            heap: sorted_iterators.flat_map(Takeble::try_new).fold(
                BinaryHeap::new(),
                |mut heap, iter| {
                    heap.push(Reverse(iter));
                    heap
                },
            ),
        }
    }
}
impl<T: Ord, I: Iterator<Item = T>> Iterator for MergeSortedIter<T, I> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(not_empty_peekable) = self.heap.pop() {
            let (el, rest) = not_empty_peekable.0.take();
            if let Some(not_empty_rest) = rest {
                self.heap.push(Reverse(not_empty_rest));
            }
            Some(el)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod merge_iterators_tests {
    use super::*;

    #[test]
    fn test_merge_sorted_iter() {
        let v1 = [1, 3, 5, 7, 9].into_iter();
        let v2 = [2, 4, 6, 8, 10].into_iter();
        let v3 = [11, 13, 15, 17, 19].into_iter();
        let v4 = [12, 14, 16, 18, 20].into_iter();

        let iter = MergeSortedIter::new([v1, v2, v3, v4].into_iter());

        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        );
    }
}
