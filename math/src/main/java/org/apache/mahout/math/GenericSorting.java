/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
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

public class GenericSorting {

  private static final int SMALL = 7;

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
}
