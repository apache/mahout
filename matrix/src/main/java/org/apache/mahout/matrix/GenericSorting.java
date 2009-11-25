/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.matrix;

import org.apache.mahout.matrix.function.IntComparator;
/**
 Generically sorts arbitrary shaped data (for example multiple arrays, 1,2 or 3-d matrices, and so on) using a
 quicksort or mergesort. This class addresses two problems, namely
 <ul>
 <li><i>Sorting multiple arrays in sync</i>
 <li><i>Sorting by multiple sorting criteria</i> (primary, secondary, tertiary,
 ...)
 </ul>
 <h4>Sorting multiple arrays in sync</h4>
 <p>
 Assume we have three arrays X, Y and Z. We want to sort all three arrays by
 X (or some arbitrary comparison function). For example, we have<br>
 <tt>X=[3, 2, 1], Y=[3.0, 2.0, 1.0], Z=[6.0, 7.0, 8.0]</tt>. The output should
 be <tt><br>
 X=[1, 2, 3], Y=[1.0, 2.0, 3.0], Z=[8.0, 7.0, 6.0]</tt>. </p>
 <p>How can we achive this? Here are several alternatives. We could ... </p>
 <ol>
 <li> make a list of Point3D objects, sort the list as desired using a comparison
 function, then copy the results back into X, Y and Z. The classic object-oriented
 way. </li>
 <li>make an index list [0,1,2,...,N-1], sort the index list using a comparison function,
 then reorder the elements of X,Y,Z as defined by the index list. Reordering
 cannot be done in-place, so we need to copy X to some temporary array, then
 copy in the right order back from the temporary into X. Same for Y and Z.
 </li>
 <li> use a generic quicksort or mergesort which, whenever two elements in X are swapped,
 also swaps the corresponding elements in Y and Z. </li>
 </ol>
 Alternatives 1 and 2 involve quite a lot of copying and allocate significant amounts
 of temporary memory. Alternative 3 involves more swapping, more polymorphic message dispatches, no copying and does not need any temporary memory.
 <p> This class implements alternative 3. It operates on arbitrary shaped data.
 In fact, it has no idea what kind of data it is sorting. Comparisons and swapping
 are delegated to user provided objects which know their data and can do the
 job.
 <p> Lets call the generic data <tt>g</tt> (it may be one array, three linked lists
 or whatever). This class takes a user comparison function operating on two indexes
 <tt>(a,b)</tt>, namely an {@link IntComparator}. The comparison function determines
 whether <tt>g[a]</tt> is equal, less or greater than <tt>g[b]</tt>. The sort,
 depending on its implementation, can decide to swap the data at index <tt>a</tt>
 with the data at index <tt>b</tt>. It calls a user provided {@link org.apache.mahout.matrix.Swapper}
 object that knows how to swap the data of these indexes.
 <p>The following snippet shows how to solve the problem.
 <table>
 <td class="PRE">
 <pre>
 final int[] x;
 final double[] y;
 final double[] z;

 x = new int[]    {3,   2,   1  };
 y = new double[] {3.0, 2.0, 1.0};
 z = new double[] {6.0, 7.0, 8.0};


 // this one knows how to swap two indexes (a,b)
 Swapper swapper = new Swapper() {
 &nbsp;&nbsp;&nbsp;public void swap(int a, int b) {
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;int t1;  double t2, t3;
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;t1 = x[a]; x[a] = x[b];  x[b] = t1;
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;t2 = y[a]; y[a] = y[b]; y[b] = t2;
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;t3 = z[a]; z[a] = z[b];  z[b] = t3;
 &nbsp;&nbsp;&nbsp;}
 };
 // simple comparison: compare by X and ignore Y,Z<br>
 IntComparator comp = new IntComparator() {
 &nbsp;&nbsp;&nbsp;public int compare(int a, int b) {
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return x[a]==x[b] ? 0 : (x[a]&lt;x[b] ? -1 : 1);
 &nbsp;&nbsp;&nbsp;}
 };

 System.out.println("before:");
 System.out.println("X="+Arrays.toString(x));
 System.out.println("Y="+Arrays.toString(y));
 System.out.println("Z="+Arrays.toString(z));

 GenericSorting.quickSort(0, X.length, comp, swapper);
 // GenericSorting.mergeSort(0, X.length, comp, swapper);

 System.out.println("after:");
 System.out.println("X="+Arrays.toString(x));
 System.out.println("Y="+Arrays.toString(y));
 System.out.println("Z="+Arrays.toString(z));
 </pre>
 </td>
 </table>
 <h4>Sorting by multiple sorting criterias (primary, secondary, tertiary, ...)</h4>
 <p>Assume again we have three arrays X, Y and Z. Now we want to sort all three
 arrays, primarily by Y, secondarily by Z (if Y elements are equal). For example,
 we have<br>
 <tt>X=[6, 7, 8, 9], Y=[3.0, 2.0, 1.0, 3.0], Z=[5.0, 4.0, 4.0, 1.0]</tt>. The
 output should be <tt><br>
 X=[8, 7, 9, 6], Y=[1.0, 2.0, 3.0, 3.0], Z=[4.0, 4.0, 1.0, 5.0]</tt>. </p>
 <p>Here is how to solve the problem. All code in the above example stays the same,
 except that we modify the comparison function as follows</p>
 <table>
 <td class="PRE">
 <pre>
 //compare by Y, if that doesn't help, reside to Z
 IntComparator comp = new IntComparator() {
 &nbsp;&nbsp;&nbsp;public int compare(int a, int b) {
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if (y[a]==y[b]) return z[a]==z[b] ? 0 : (z[a]&lt;z[b] ? -1 : 1);
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return y[a]&lt;y[b] ? -1 : 1;
 &nbsp;&nbsp;&nbsp;}
 };
 </pre>
 </td>
 </table>

 <h4>Notes</h4>
 <p></p>
 <p> Sorts involving floating point data and not involving comparators, like, for
 example provided in the JDK {@link java.util.Arrays} and in the Colt {@link
org.apache.mahout.matrix.Sorting} handle floating point numbers in special ways to guarantee
 that NaN's are swapped to the end and -0.0 comes before 0.0. Methods delegating
 to comparators cannot do this. They rely on the comparator. Thus, if such boundary
 cases are an issue for the application at hand, comparators explicitly need
 to implement -0.0 and NaN aware comparisons. Remember: <tt>-0.0 < 0.0 == false</tt>,
 <tt>(-0.0 == 0.0) == true</tt>, as well as <tt>5.0 &lt; Double.NaN == false</tt>,
 <tt>5.0 &gt; Double.NaN == false</tt>. Same for <tt>float</tt>.
 <h4>Implementation </h4>
 <p>The quicksort is a derivative of the JDK 1.2 V1.26 algorithms (which are, in
 turn, based on Bentley's and McIlroy's fine work).
 The mergesort is a derivative of the JAL algorithms, with optimisations taken from the JDK algorithms.
 Both quick and merge sort are "in-place", i.e. do not allocate temporary memory (helper arrays).
 Mergesort is <i>stable</i> (by definition), while quicksort is not.
 A stable sort is, for example, helpful, if matrices are sorted successively
 by multiple columns. It preserves the relative position of equal elements.

 @see java.util.Arrays
 @see org.apache.mahout.matrix.Sorting
 @see org.apache.mahout.matrix.matrix.doublealgo.Sorting

 @author wolfgang.hoschek@cern.ch
 @version 1.0, 03-Jul-99
 */

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
   * @param array Array containing the range.
   * @param first Beginning of the range.
   * @param last  One past the end of the range.
   * @param x     Element to be searched for.
   * @param comp  Comparison function.
   * @return The largest index i such that, for every j in the range <code>[first, i)</code>, <code>comp.apply(array[j],
   *         x)</code> is <code>true</code>.
   * @see Sorting#upper_bound
   * @see Sorting#equal_range
   * @see Sorting#binary_search
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
   * @param array Array containing the sequence
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
   * @param array  Array containing the range
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
   * @param array Array containing the range.
   * @param first Beginning of the range.
   * @param last  One past the end of the range.
   * @param x     Element to be searched for.
   * @param comp  Comparison function.
   * @return The largest index i such that, for every j in the range <code>[first, i)</code>, <code>comp.apply(x,
   *         array[j])</code> is <code>false</code>.
   * @see Sorting#lower_bound
   * @see Sorting#equal_range
   * @see Sorting#binary_search
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
