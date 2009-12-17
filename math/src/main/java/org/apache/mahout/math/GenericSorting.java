/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math;

import org.apache.mahout.math.function.IntComparator;

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class GenericSorting {

  private static final int SMALL = 7;
  private static final int MEDIUM = 40;

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private GenericSorting() {
  }

  /**
   * Transforms two consecutive sorted ranges into a single sorted range.  The initial ranges are <code>[first,
   * middle)</code> and <code>[middle, last)</code>, and the resulting range is <code>[first, last)</code>. Elements in
   * the first input range will precede equal elements in the second.
   */
  private static void inplace_merge(int first, int middle, int last, IntComparator comp, Swapper swapper) {
    if (first >= middle || middle >= last) {
      return;
    }
    if (last - first == 2) {
      if (comp.compare(middle, first) < 0) {
        swapper.swap(first, middle);
      }
      return;
    }
    int firstCut;
    int secondCut;
    if (middle - first > last - middle) {
      firstCut = first + (middle - first) / 2;
      secondCut = lower_bound(middle, last, firstCut, comp);
    } else {
      secondCut = middle + (last - middle) / 2;
      firstCut = upper_bound(first, middle, secondCut, comp);
    }

    // rotate(firstCut, middle, secondCut, swapper);
    // is manually inlined for speed (jitter inlining seems to work only for small call depths, even if methods are "static private")
    // speedup = 1.7
    // begin inline
    int first2 = firstCut;
    int middle2 = middle;
    int last2 = secondCut;
    if (middle2 != first2 && middle2 != last2) {
      int first1 = first2;
      int last1 = middle2;
      while (first1 < --last1) {
        swapper.swap(first1++, last1);
      }
      first1 = middle2;
      last1 = last2;
      while (first1 < --last1) {
        swapper.swap(first1++, last1);
      }
      first1 = first2;
      last1 = last2;
      while (first1 < --last1) {
        swapper.swap(first1++, last1);
      }
    }
    // end inline

    middle = firstCut + (secondCut - middle);
    inplace_merge(first, firstCut, middle, comp, swapper);
    inplace_merge(middle, secondCut, last, comp, swapper);
  }

  /**
   * Performs a binary search on an already-sorted range: finds the first position where an element can be inserted
   * without violating the ordering. Sorting is by a user-supplied comparison function.
   *
   * @param first Beginning of the range.
   * @param last  One past the end of the range.
   * @param x     Element to be searched for.
   * @param comp  Comparison function.
   * @return The largest index i such that, for every j in the range <code>[first, i)</code>, <code>comp.apply(array[j],
   *         x)</code> is <code>true</code>.
   * @see Sorting#upper_bound
   */
  private static int lower_bound(int first, int last, int x, IntComparator comp) {
    //if (comp==null) throw new NullPointerException();
    int len = last - first;
    while (len > 0) {
      int half = len / 2;
      int middle = first + half;
      if (comp.compare(middle, x) < 0) {
        first = middle + 1;
        len -= half + 1;
      } else {
        len = half;
      }
    }
    return first;
  }

  /** Returns the index of the median of the three indexed chars. */
  private static int med3(int a, int b, int c, IntComparator comp) {
    int ab = comp.compare(a, b);
    int ac = comp.compare(a, c);
    int bc = comp.compare(b, c);
    return (ab < 0 ?
        (bc < 0 ? b : ac < 0 ? c : a) :
        (bc > 0 ? b : ac > 0 ? c : a));
  }

  /**
   * Sorts the specified range of elements according to the order induced by the specified comparator.  All elements in
   * the range must be <i>mutually comparable</i> by the specified comparator (that is, <tt>c.compare(a, b)</tt> must
   * not throw an exception for any indexes <tt>a</tt> and <tt>b</tt> in the range).<p>
   *
   * This sort is guaranteed to be <i>stable</i>:  equal elements will not be reordered as a result of the sort.<p>
   *
   * The sorting algorithm is a modified mergesort (in which the merge is omitted if the highest element in the low
   * sublist is less than the lowest element in the high sublist).  This algorithm offers guaranteed n*log(n)
   * performance, and can approach linear performance on nearly sorted lists.
   *
   * @param fromIndex the index of the first element (inclusive) to be sorted.
   * @param toIndex   the index of the last element (exclusive) to be sorted.
   * @param c         the comparator to determine the order of the generic data.
   * @param swapper   an object that knows how to swap the elements at any two indexes (a,b).
   * @see IntComparator
   * @see Swapper
   */
  public static void mergeSort(int fromIndex, int toIndex, IntComparator c, Swapper swapper) {
    /*
      We retain the same method signature as quickSort.
      Given only a comparator and swapper we do not know how to copy and move elements from/to temporary arrays.
      Hence, in contrast to the JDK mergesorts this is an "in-place" mergesort, i.e. does not allocate any temporary arrays.
      A non-inplace mergesort would perhaps be faster in most cases, but would require non-intuitive delegate objects...
    */
    int length = toIndex - fromIndex;

    // Insertion sort on smallest arrays
    if (length < SMALL) {
      for (int i = fromIndex; i < toIndex; i++) {
        for (int j = i; j > fromIndex && (c.compare(j - 1, j) > 0); j--) {
          swapper.swap(j, j - 1);
        }
      }
      return;
    }

    // Recursively sort halves
    int mid = (fromIndex + toIndex) / 2;
    mergeSort(fromIndex, mid, c, swapper);
    mergeSort(mid, toIndex, c, swapper);

    // If list is already sorted, nothing left to do.  This is an
    // optimization that results in faster sorts for nearly ordered lists.
    if (c.compare(mid - 1, mid) <= 0) {
      return;
    }

    // Merge sorted halves
    inplace_merge(fromIndex, mid, toIndex, c, swapper);
  }

  /**
   * Sorts the specified range of elements according to the order induced by the specified comparator.  All elements in
   * the range must be <i>mutually comparable</i> by the specified comparator (that is, <tt>c.compare(a, b)</tt> must
   * not throw an exception for any indexes <tt>a</tt> and <tt>b</tt> in the range).<p>
   *
   * The sorting algorithm is a tuned quicksort, adapted from Jon L. Bentley and M. Douglas McIlroy's "Engineering a
   * Sort Function", Software-Practice and Experience, Vol. 23(11) P. 1249-1265 (November 1993).  This algorithm offers
   * n*log(n) performance on many data sets that cause other quicksorts to degrade to quadratic performance.
   *
   * @param fromIndex the index of the first element (inclusive) to be sorted.
   * @param toIndex   the index of the last element (exclusive) to be sorted.
   * @param c         the comparator to determine the order of the generic data.
   * @param swapper   an object that knows how to swap the elements at any two indexes (a,b).
   * @see IntComparator
   * @see Swapper
   */
  public static void quickSort(int fromIndex, int toIndex, IntComparator c, Swapper swapper) {
    quickSort1(fromIndex, toIndex - fromIndex, c, swapper);
  }

  /** Sorts the specified sub-array into ascending order. */
  private static void quickSort1(int off, int len, IntComparator comp, Swapper swapper) {
    // Insertion sort on smallest arrays
    if (len < SMALL) {
      for (int i = off; i < len + off; i++) {
        for (int j = i; j > off && (comp.compare(j - 1, j) > 0); j--) {
          swapper.swap(j, j - 1);
        }
      }
      return;
    }

    // Choose a partition element, v
    int m = off + len / 2;       // Small arrays, middle element
    if (len > SMALL) {
      int l = off;
      int n = off + len - 1;
      if (len > MEDIUM) {        // Big arrays, pseudomedian of 9
        int s = len / 8;
        l = med3(l, l + s, l + 2 * s, comp);
        m = med3(m - s, m, m + s, comp);
        n = med3(n - 2 * s, n - s, n, comp);
      }
      m = med3(l, m, n, comp); // Mid-size, med of 3
    }
    //long v = x[m];

    // Establish Invariant: v* (<v)* (>v)* v*
    int a = off, b = a, c = off + len - 1, d = c;
    while (true) {
      int comparison;
      while (b <= c && ((comparison = comp.compare(b, m)) <= 0)) {
        if (comparison == 0) {
          if (a == m) {
            m = b; // moving target; DELTA to JDK !!!
          } else if (b == m) {
            m = a;
          } // moving target; DELTA to JDK !!!
          swapper.swap(a++, b);
        }
        b++;
      }
      while (c >= b && ((comparison = comp.compare(c, m)) >= 0)) {
        if (comparison == 0) {
          if (c == m) {
            m = d; // moving target; DELTA to JDK !!!
          } else if (d == m) {
            m = c;
          } // moving target; DELTA to JDK !!!
          swapper.swap(c, d--);
        }
        c--;
      }
      if (b > c) {
        break;
      }
      if (b == m) {
        m = d; // moving target; DELTA to JDK !!!
      } else if (c == m) {
        m = c;
      } // moving target; DELTA to JDK !!!
      swapper.swap(b++, c--);
    }

    // Swap partition elements back to middle
    int n = off + len;
    int s = Math.min(a - off, b - a);
    vecswap(swapper, off, b - s, s);
    s = Math.min(d - c, n - d - 1);
    vecswap(swapper, b, n - s, s);

    // Recursively sort non-partition-elements
    if ((s = b - a) > 1) {
      quickSort1(off, s, comp, swapper);
    }
    if ((s = d - c) > 1) {
      quickSort1(n - s, s, comp, swapper);
    }
  }

  /**
   * Reverses a sequence of elements.
   *
   * @param first Beginning of the range
   * @param last  One past the end of the range
   * @throws ArrayIndexOutOfBoundsException If the range is invalid.
   */
  private static void reverse(int first, int last, Swapper swapper) {
    // no more needed since manually inlined
    while (first < --last) {
      swapper.swap(first++, last);
    }
  }

  /**
   * Rotate a range in place: <code>array[middle]</code> is put in <code>array[first]</code>,
   * <code>array[middle+1]</code> is put in <code>array[first+1]</code>, etc.  Generally, the element in position
   * <code>i</code> is put into position <code>(i + (last-middle)) % (last-first)</code>.
   *
   * @param first  Beginning of the range
   * @param middle Index of the element that will be put in <code>array[first]</code>
   * @param last   One past the end of the range
   */
  private static void rotate(int first, int middle, int last, Swapper swapper) {
    // no more needed since manually inlined
    if (middle != first && middle != last) {
      reverse(first, middle, swapper);
      reverse(middle, last, swapper);
      reverse(first, last, swapper);
    }
  }

  /**
   * Performs a binary search on an already-sorted range: finds the last position where an element can be inserted
   * without violating the ordering. Sorting is by a user-supplied comparison function.
   *
   * @param first Beginning of the range.
   * @param last  One past the end of the range.
   * @param x     Element to be searched for.
   * @param comp  Comparison function.
   * @return The largest index i such that, for every j in the range <code>[first, i)</code>, <code>comp.apply(x,
   *         array[j])</code> is <code>false</code>.
   * @see Sorting#lower_bound
   */
  private static int upper_bound(int first, int last, int x, IntComparator comp) {
    //if (comp==null) throw new NullPointerException();
    int len = last - first;
    while (len > 0) {
      int half = len / 2;
      int middle = first + half;
      if (comp.compare(x, middle) < 0) {
        len = half;
      } else {
        first = middle + 1;
        len -= half + 1;
      }
    }
    return first;
  }

  /** Swaps x[a .. (a+n-1)] with x[b .. (b+n-1)]. */
  private static void vecswap(Swapper swapper, int a, int b, int n) {
    for (int i = 0; i < n; i++, a++, b++) {
      swapper.swap(a, b);
    }
  }
}
