//! Shared masked descriptor traversal for contiguous edge lines and strided
//! columns of an edge grid. Only adjacent equal descriptors share lanes.

#[derive(Clone, Copy)]
pub(crate) struct Descriptors<'a, T> {
    pub entries: &'a [Option<T>],
    pub stride: usize,
    pub count: usize,
}

impl<'a, T: Copy + Eq> Descriptors<'a, T> {
    #[inline]
    pub fn runs(self) -> Runs<'a, T> {
        Runs {
            descriptors: self,
            next: 0,
        }
    }
}

pub(crate) struct Runs<'a, T> {
    descriptors: Descriptors<'a, T>,
    next: usize,
}

impl<T: Copy + Eq> Iterator for Runs<'_, T> {
    type Item = (usize, usize, T);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let d = &self.descriptors;
        while self.next < d.count {
            let Some(filter) = d.entries[self.next * d.stride] else {
                self.next += 1;
                continue;
            };
            let first = self.next;
            self.next += 1;
            while self.next < d.count && d.entries[self.next * d.stride] == Some(filter) {
                self.next += 1;
            }
            return Some((first, self.next - first, filter));
        }
        None
    }
}
