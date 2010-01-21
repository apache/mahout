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
import org.apache.mahout.math.list.DoubleArrayList;
import org.apache.mahout.math.list.IntArrayList;

import java.util.Comparator;
/**
 * Given some interval boundaries, partitions arrays such that all elements falling into an interval are placed next to each other.
 * <p>
 * The algorithms partition arrays into two or more intervals. 
 * They distinguish between <i>synchronously</i> partitioning either one, two or three arrays.
 * They further come in templated versions, either partitioning <tt>int[]</tt> arrays or <tt>double[]</tt> arrays.
 * <p>
 * You may want to start out reading about the simplest case: Partitioning one <tt>int[]</tt> array into two intervals.
 * To do so, read {@link #partition(int[],int,int,int)}.
 *
 * Next, building upon that foundation comes a method partitioning <tt>int[]</tt> arrays into multiple intervals.
 * See {@link #partition(int[],int,int,int[],int,int,int[])} for related documentation.
 * <p>
 * All other methods are no different than the one's you now already understand, except that they operate on slightly different data types.
 * <p>
 * <b>Performance</b>
 * <p>
 * Partitioning into two intervals is <tt>O( N )</tt>.
 * Partitioning into k intervals is <tt>O( N * log(k))</tt>.
 * Constants factors are minimized.
 * No temporary memory is allocated; Partitioning is in-place.
 *
 * @see org.apache.mahout.math.matrix.doublealgo.Partitioning
 *
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class Partitioning {

  private static final int SMALL = 7;
  private static final int MEDIUM = 40;

  // benchmark only
  protected static int steps = 0;
  //public static int swappedElements = 0;

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private Partitioning() {
  }

  /**
   * Finds the given key "a" within some generic data using the binary search algorithm.
   *
   * @param a    the index of the key to search for.
   * @param from the leftmost search position, inclusive.
   * @param to   the rightmost search position, inclusive.
   * @param comp the comparator determining the order of the generic data. Takes as first argument the index <tt>a</tt>
   *             within the generic splitters <tt>s</tt>. Takes as second argument the index <tt>b</tt> within the
   *             generic data <tt>g</tt>.
   * @return index of the search key, if it is contained in the list; otherwise, <tt>(-(<i>insertion point</i>) -
   *         1)</tt>. The <i>insertion point</i> is defined as the the point at which the value would be inserted into
   *         the list: the index of the first element greater than the key, or <tt>list.length</tt>, if all elements in
   *         the list are less than the specified key.  Note that this guarantees that the return value will be &gt;= 0
   *         if and only if the key is found.
   */
  private static int binarySearchFromTo(int a, int from, int to, IntComparator comp) {
    while (from <= to) {
      int mid = (from + to) / 2;
      int comparison = comp.compare(mid, a);
      if (comparison < 0) {
        from = mid + 1;
      } else if (comparison > 0) {
        to = mid - 1;
      } else {
        return mid;
      } // key found
    }
    return -(from + 1);  // key not found.
  }

  /**
   * Same as {@link #dualPartition(int[],int[],int,int,int[],int,int,int[])} except that it <i>synchronously</i>
   * partitions <tt>double[]</tt> rather than <tt>int[]</tt> arrays.
   */
  public static void dualPartition(double[] list, double[] secondary, int from, int to, double[] splitters,
                                   int splitFrom, int splitTo, int[] splitIndexes) {

    if (splitFrom > splitTo) {
      return;
    } // nothing to do
    if (from > to) { // all bins are empty
      from--;
      for (int i = splitFrom; i <= splitTo;) {
        splitIndexes[i++] = from;
      }
      return;
    }

    // Choose a partition (pivot) index, m
    // Ideally, the pivot should be the median, because a median splits a list into two equal sized sublists.
    // However, computing the median is expensive, so we use an approximation.
    int medianIndex;
    if (splitFrom == splitTo) { // we don't really have a choice
      medianIndex = splitFrom;
    } else { // we do have a choice
      int m = (from + to) / 2;       // Small arrays, middle element
      int len = to - from + 1;
      if (len > SMALL) {
        int l = from;
        int n = to;
        if (len > MEDIUM) {        // Big arrays, pseudomedian of 9
          int s = len / 8;
          l = med3(list, l, l + s, l + 2 * s);
          m = med3(list, m - s, m, m + s);
          n = med3(list, n - 2 * s, n - s, n);
        }
        m = med3(list, l, m, n); // Mid-size, pseudomedian of 3
      }

      // Find the splitter closest to the pivot, i.e. the splitter that best splits the list into two equal sized sublists.
      medianIndex = Sorting.binarySearchFromTo(splitters, list[m], splitFrom, splitTo);
      if (medianIndex < 0) {
        medianIndex = -medianIndex - 1;
      } // not found
      if (medianIndex > splitTo) {
        medianIndex = splitTo;
      } // not found, one past the end

    }
    double splitter = splitters[medianIndex]; // int, double --> template type dependent

    // Partition the list according to the splitter, i.e.
    // Establish invariant: list[i] < splitter <= list[j] for i=from..medianIndex and j=medianIndex+1 .. to
    int splitIndex = dualPartition(list, secondary, from, to, splitter);
    splitIndexes[medianIndex] = splitIndex;

    // Optimization: Handle special cases to cut down recursions.
    if (splitIndex < from) { // no element falls into this bin
      // all bins with splitters[i] <= splitter are empty
      int i = medianIndex - 1;
      while (i >= splitFrom && (!(splitter < splitters[i]))) {
        splitIndexes[i--] = splitIndex;
      }
      splitFrom = medianIndex + 1;
    } else if (splitIndex >= to) { // all elements fall into this bin
      // all bins with splitters[i] >= splitter are empty
      int i = medianIndex + 1;
      while (i <= splitTo && (!(splitter > splitters[i]))) {
        splitIndexes[i++] = splitIndex;
      }
      splitTo = medianIndex - 1;
    }

    // recursively partition left half
    if (splitFrom <= medianIndex - 1) {
      dualPartition(list, secondary, from, splitIndex, splitters, splitFrom, medianIndex - 1, splitIndexes);
    }

    // recursively partition right half
    if (medianIndex + 1 <= splitTo) {
      dualPartition(list, secondary, splitIndex + 1, to, splitters, medianIndex + 1, splitTo, splitIndexes);
    }
  }

  /**
   * Same as {@link #dualPartition(int[],int[],int,int,int)} except that it <i>synchronously</i> partitions
   * <tt>double[]</tt> rather than <tt>int[]</tt> arrays.
   */
  public static int dualPartition(double[] list, double[] secondary, int from, int to, double splitter) {
    for (int i = from - 1; ++i <= to;) {
      double element = list[i];  // int, double --> template type dependent
      if (element < splitter) {
        // swap x[i] with x[from]
        list[i] = list[from];
        list[from] = element;

        element = secondary[i];
        secondary[i] = secondary[from];
        secondary[from++] = element;
      }
    }
    return from - 1;
  }

  /**
   * Same as {@link #partition(int[],int,int,int[],int,int,int[])} except that this method <i>synchronously</i>
   * partitions two arrays at the same time; both arrays are partially sorted according to the elements of the primary
   * array. In other words, each time an element in the primary array is moved from index A to B, the correspoding
   * element within the secondary array is also moved from index A to B. <p> <b>Use cases:</b> <p> Image having a large
   * list of 2-dimensional points. If memory consumption and performance matter, it is a good idea to physically lay
   * them out as two 1-dimensional arrays (using something like <tt>Point2D</tt> objects would be prohibitively
   * expensive, both in terms of time and space). Now imagine wanting to histogram the points. We may want to partially
   * sort the points by x-coordinate into intervals. This method efficiently does the job. <p> <b>Performance:</b> <p>
   * Same as for single-partition methods.
   */
  public static void dualPartition(int[] list, int[] secondary, int from, int to, int[] splitters, int splitFrom,
                                   int splitTo, int[] splitIndexes) {

    if (splitFrom > splitTo) {
      return;
    } // nothing to do
    if (from > to) { // all bins are empty
      from--;
      for (int i = splitFrom; i <= splitTo;) {
        splitIndexes[i++] = from;
      }
      return;
    }

    // Choose a partition (pivot) index, m
    // Ideally, the pivot should be the median, because a median splits a list into two equal sized sublists.
    // However, computing the median is expensive, so we use an approximation.
    int medianIndex;
    if (splitFrom == splitTo) { // we don't really have a choice
      medianIndex = splitFrom;
    } else { // we do have a choice
      int m = (from + to) / 2;       // Small arrays, middle element
      int len = to - from + 1;
      if (len > SMALL) {
        int l = from;
        int n = to;
        if (len > MEDIUM) {        // Big arrays, pseudomedian of 9
          int s = len / 8;
          l = med3(list, l, l + s, l + 2 * s);
          m = med3(list, m - s, m, m + s);
          n = med3(list, n - 2 * s, n - s, n);
        }
        m = med3(list, l, m, n); // Mid-size, pseudomedian of 3
      }

      // Find the splitter closest to the pivot, i.e. the splitter that best splits the list into two equal sized sublists.
      medianIndex = Sorting.binarySearchFromTo(splitters, list[m], splitFrom, splitTo);
      if (medianIndex < 0) {
        medianIndex = -medianIndex - 1;
      } // not found
      if (medianIndex > splitTo) {
        medianIndex = splitTo;
      } // not found, one past the end

    }
    int splitter = splitters[medianIndex]; // int, double --> template type dependent

    // Partition the list according to the splitter, i.e.
    // Establish invariant: list[i] < splitter <= list[j] for i=from..medianIndex and j=medianIndex+1 .. to
    int splitIndex = dualPartition(list, secondary, from, to, splitter);
    splitIndexes[medianIndex] = splitIndex;

    // Optimization: Handle special cases to cut down recursions.
    if (splitIndex < from) { // no element falls into this bin
      // all bins with splitters[i] <= splitter are empty
      int i = medianIndex - 1;
      while (i >= splitFrom && (!(splitter < splitters[i]))) {
        splitIndexes[i--] = splitIndex;
      }
      splitFrom = medianIndex + 1;
    } else if (splitIndex >= to) { // all elements fall into this bin
      // all bins with splitters[i] >= splitter are empty
      int i = medianIndex + 1;
      while (i <= splitTo && (!(splitter > splitters[i]))) {
        splitIndexes[i++] = splitIndex;
      }
      splitTo = medianIndex - 1;
    }

    // recursively partition left half
    if (splitFrom <= medianIndex - 1) {
      dualPartition(list, secondary, from, splitIndex, splitters, splitFrom, medianIndex - 1, splitIndexes);
    }

    // recursively partition right half
    if (medianIndex + 1 <= splitTo) {
      dualPartition(list, secondary, splitIndex + 1, to, splitters, medianIndex + 1, splitTo, splitIndexes);
    }
  }

  /**
   * Same as {@link #partition(int[],int,int,int)} except that this method <i>synchronously</i> partitions two arrays at
   * the same time; both arrays are partially sorted according to the elements of the primary array. In other words,
   * each time an element in the primary array is moved from index A to B, the correspoding element within the secondary
   * array is also moved from index A to B. <p> <b>Performance:</b> <p> Same as for single-partition methods.
   */
  public static int dualPartition(int[] list, int[] secondary, int from, int to, int splitter) {
    for (int i = from - 1; ++i <= to;) {
      int element = list[i];  // int, double --> template type dependent
      if (element < splitter) {
        // swap x[i] with x[from]
        list[i] = list[from];
        list[from] = element;

        element = secondary[i];
        secondary[i] = secondary[from];
        secondary[from++] = element;
      }
    }
    return from - 1;
  }

  /**
   * Same as {@link #partition(int[],int,int,int[],int,int,int[])} except that it <i>generically</i> partitions
   * arbitrary shaped data (for example matrices or multiple arrays) rather than <tt>int[]</tt> arrays. <p> This method
   * operates on arbitrary shaped data and arbitrary shaped splitters. In fact, it has no idea what kind of data by what
   * kind of splitters it is partitioning. Comparisons and swapping are delegated to user provided objects which know
   * their data and can do the job. <p> Lets call the generic data <tt>g</tt> (it may be a matrix, one array, three
   * linked lists or whatever). Lets call the generic splitters <tt>s</tt>. This class takes a user comparison function
   * operating on two indexes <tt>(a,b)</tt>, namely an {@link IntComparator}. The comparison function determines
   * whether <tt>s[a]</tt> is equal, less or greater than <tt>g[b]</tt>. This method can then decide to swap the data
   * <tt>g[b]</tt> with the data <tt>g[c]</tt> (yes, <tt>c</tt>, not <tt>a</tt>). It calls a user provided {@link
   * org.apache.mahout.math.Swapper} object that knows how to swap the data of these two indexes. <p> Again, note the
   * details: Comparisons compare <tt>s[a]</tt> with <tt>g[b]</tt>. Swaps swap <tt>g[b]</tt> with <tt>g[c]</tt>. Prior
   * to calling this method, the generic splitters <tt>s</tt> must be sorted ascending and must not contain multiple
   * equal values. These preconditions are not checked; be sure that they are met.
   *
   * @param from         the index of the first element within <tt>g</tt> to be considered.
   * @param to           the index of the last element within <tt>g</tt> to be considered. The method considers the
   *                     elements <tt>g[from] .. g[to]</tt>.
   * @param splitFrom    the index of the first splitter element to be considered.
   * @param splitTo      the index of the last splitter element to be considered. The method considers the splitter
   *                     elements <tt>s[splitFrom] .. s[splitTo]</tt>.
   * @param splitIndexes a list into which this method fills the indexes of elements delimiting intervals. Upon return
   *                     <tt>splitIndexes[splitFrom..splitTo]</tt> will be set accordingly. Therefore, must satisfy
   *                     <tt>splitIndexes.length > splitTo</tt>.
   * @param comp         the comparator comparing a splitter with an element of the generic data. Takes as first
   *                     argument the index <tt>a</tt> within the generic splitters <tt>s</tt>. Takes as second argument
   *                     the index <tt>b</tt> within the generic data <tt>g</tt>.
   * @param comp2        the comparator to determine the order of the generic data. Takes as first argument the index
   *                     <tt>a</tt> within the generic data <tt>g</tt>. Takes as second argument the index <tt>b</tt>
   *                     within the generic data <tt>g</tt>.
   * @param comp3        the comparator comparing a splitter with another splitter. Takes as first argument the index
   *                     <tt>a</tt> within the generic splitters <tt>s</tt>. Takes as second argument the index
   *                     <tt>b</tt> within the generic splitters <tt>g</tt>.
   * @param swapper      an object that knows how to swap the elements at any two indexes (a,b). Takes as first argument
   *                     the index <tt>b</tt> within the generic data <tt>g</tt>. Takes as second argument the index
   *                     <tt>c</tt> within the generic data <tt>g</tt>.
   *
   *                     <p> Tip: Normally you will have <tt>splitIndexes.length == s.length</tt> as well as
   *                     <tt>from==0, to==g.length-1</tt> and <tt>splitFrom==0, splitTo==s.length-1</tt>.
   * @see Sorting#binarySearchFromTo(int,int,IntComparator)
   */
  public static void genericPartition(int from, int to, int splitFrom, int splitTo, int[] splitIndexes,
                                      IntComparator comp, IntComparator comp2, IntComparator comp3, Swapper swapper) {

    if (splitFrom > splitTo) {
      return;
    } // nothing to do
    if (from > to) { // all bins are empty
      from--;
      for (int i = splitFrom; i <= splitTo;) {
        splitIndexes[i++] = from;
      }
      return;
    }

    // Choose a partition (pivot) index, m
    // Ideally, the pivot should be the median, because a median splits a list into two equal sized sublists.
    // However, computing the median is expensive, so we use an approximation.
    int medianIndex;
    if (splitFrom == splitTo) { // we don't really have a choice
      medianIndex = splitFrom;
    } else { // we do have a choice
      int m = (from + to) / 2;       // Small arrays, middle element
      int len = to - from + 1;
      if (len > SMALL) {
        int l = from;
        int n = to;
        if (len > MEDIUM) {        // Big arrays, pseudomedian of 9
          int s = len / 8;
          l = med3(l, l + s, l + 2 * s, comp2);
          m = med3(m - s, m, m + s, comp2);
          n = med3(n - 2 * s, n - s, n, comp2);
        }
        m = med3(l, m, n, comp2); // Mid-size, pseudomedian of 3
      }

      // Find the splitter closest to the pivot, i.e. the splitter that best splits the list into two equal sized sublists.
      medianIndex = binarySearchFromTo(m, splitFrom, splitTo, comp);
      if (medianIndex < 0) {
        medianIndex = -medianIndex - 1;
      } // not found
      if (medianIndex > splitTo) {
        medianIndex = splitTo;
      } // not found, one past the end

    }
    int splitter = medianIndex; // int, double --> template type dependent

    // Partition the list according to the splitter, i.e.
    // Establish invariant: list[i] < splitter <= list[j] for i=from..medianIndex and j=medianIndex+1 .. to
    int splitIndex = genericPartition(from, to, splitter, comp, swapper);
    splitIndexes[medianIndex] = splitIndex;


    // Optimization: Handle special cases to cut down recursions.
    if (splitIndex < from) { // no element falls into this bin
      // all bins with splitters[i] <= splitter are empty
      int i = medianIndex - 1;
      while (i >= splitFrom && (!(comp3.compare(splitter, i) < 0))) {
        splitIndexes[i--] = splitIndex;
      }
      splitFrom = medianIndex + 1;
    } else if (splitIndex >= to) { // all elements fall into this bin
      // all bins with splitters[i] >= splitter are empty
      int i = medianIndex + 1;
      while (i <= splitTo && (!(comp3.compare(splitter, i) > 0))) {
        splitIndexes[i++] = splitIndex;
      }
      splitTo = medianIndex - 1;
    }


    // recursively partition left half
    if (splitFrom <= medianIndex - 1) {
      genericPartition(from, splitIndex, splitFrom, medianIndex - 1, splitIndexes, comp, comp2, comp3, swapper);
    }

    // recursively partition right half
    if (medianIndex + 1 <= splitTo) {
      genericPartition(splitIndex + 1, to, medianIndex + 1, splitTo, splitIndexes, comp, comp2, comp3, swapper);
    }
  }

  /**
   * Same as {@link #partition(int[],int,int,int)} except that it <i>generically</i> partitions arbitrary shaped data
   * (for example matrices or multiple arrays) rather than <tt>int[]</tt> arrays.
   */
  private static int genericPartition(int from, int to, int splitter, IntComparator comp, Swapper swapper) {
    for (int i = from - 1; ++i <= to;) {
      if (comp.compare(splitter, i) > 0) {
        // swap x[i] with x[from]
        swapper.swap(i, from);
        from++;
      }
    }
    return from - 1;
  }

  /** Returns the index of the median of the three indexed elements. */
  private static int med3(double[] x, int a, int b, int c) {
    return (x[a] < x[b] ?
        (x[b] < x[c] ? b : x[a] < x[c] ? c : a) :
        (x[b] > x[c] ? b : x[a] > x[c] ? c : a));
  }

  /** Returns the index of the median of the three indexed elements. */
  private static int med3(int[] x, int a, int b, int c) {
    return (x[a] < x[b] ?
        (x[b] < x[c] ? b : x[a] < x[c] ? c : a) :
        (x[b] > x[c] ? b : x[a] > x[c] ? c : a));
  }

  /** Returns the index of the median of the three indexed chars. */
  private static int med3(Object[] x, int a, int b, int c, Comparator<Object> comp) {
    int ab = comp.compare(x[a], x[b]);
    int ac = comp.compare(x[a], x[c]);
    int bc = comp.compare(x[b], x[c]);
    return (ab < 0 ?
        (bc < 0 ? b : ac < 0 ? c : a) :
        (bc > 0 ? b : ac > 0 ? c : a));
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
   * Same as {@link #partition(int[],int,int,int[],int,int,int[])} except that it partitions <tt>double[]</tt> rather
   * than <tt>int[]</tt> arrays.
   */
  public static void partition(double[] list, int from, int to, double[] splitters, int splitFrom, int splitTo,
                               int[] splitIndexes) {

    if (splitFrom > splitTo) {
      return;
    } // nothing to do
    if (from > to) { // all bins are empty
      from--;
      for (int i = splitFrom; i <= splitTo;) {
        splitIndexes[i++] = from;
      }
      return;
    }

    // Choose a partition (pivot) index, m
    // Ideally, the pivot should be the median, because a median splits a list into two equal sized sublists.
    // However, computing the median is expensive, so we use an approximation.
    int medianIndex;
    if (splitFrom == splitTo) { // we don't really have a choice
      medianIndex = splitFrom;
    } else { // we do have a choice
      int m = (from + to) / 2;       // Small arrays, middle element
      int len = to - from + 1;
      if (len > SMALL) {
        int l = from;
        int n = to;
        if (len > MEDIUM) {        // Big arrays, pseudomedian of 9
          int s = len / 8;
          l = med3(list, l, l + s, l + 2 * s);
          m = med3(list, m - s, m, m + s);
          n = med3(list, n - 2 * s, n - s, n);
        }
        m = med3(list, l, m, n); // Mid-size, pseudomedian of 3
      }

      // Find the splitter closest to the pivot, i.e. the splitter that best splits the list into two equal sized sublists.
      medianIndex = Sorting.binarySearchFromTo(splitters, list[m], splitFrom, splitTo);
      if (medianIndex < 0) {
        medianIndex = -medianIndex - 1;
      } // not found
      if (medianIndex > splitTo) {
        medianIndex = splitTo;
      } // not found, one past the end

    }
    double splitter = splitters[medianIndex]; // int, double --> template type dependent

    // Partition the list according to the splitter, i.e.
    // Establish invariant: list[i] < splitter <= list[j] for i=from..medianIndex and j=medianIndex+1 .. to
    int splitIndex = partition(list, from, to, splitter);
    splitIndexes[medianIndex] = splitIndex;

    // Optimization: Handle special cases to cut down recursions.
    if (splitIndex < from) { // no element falls into this bin
      // all bins with splitters[i] <= splitter are empty
      int i = medianIndex - 1;
      while (i >= splitFrom && (!(splitter < splitters[i]))) {
        splitIndexes[i--] = splitIndex;
      }
      splitFrom = medianIndex + 1;
    } else if (splitIndex >= to) { // all elements fall into this bin
      // all bins with splitters[i] >= splitter are empty
      int i = medianIndex + 1;
      while (i <= splitTo && (!(splitter > splitters[i]))) {
        splitIndexes[i++] = splitIndex;
      }
      splitTo = medianIndex - 1;
    }

    // recursively partition left half
    if (splitFrom <= medianIndex - 1) {
      partition(list, from, splitIndex, splitters, splitFrom, medianIndex - 1, splitIndexes);
    }

    // recursively partition right half
    if (medianIndex + 1 <= splitTo) {
      partition(list, splitIndex + 1, to, splitters, medianIndex + 1, splitTo, splitIndexes);
    }
  }

  /**
   * Same as {@link #partition(int[],int,int,int)} except that it partitions <tt>double[]</tt> rather than
   * <tt>int[]</tt> arrays.
   */
  public static int partition(double[] list, int from, int to, double splitter) {
    for (int i = from - 1; ++i <= to;) {
      double element = list[i];  // int, double --> template type dependent
      if (element < splitter) {
        // swap x[i] with x[from]
        list[i] = list[from];
        list[from++] = element;
      }
    }
    return from - 1;
  }

  /**
   * Partitions (partially sorts) the given list such that all elements falling into some intervals are placed next to
   * each other. Returns the indexes of elements delimiting intervals. <p> <b>Example:</b> <p> <tt>list = (7, 4, 5, 50,
   * 6, 4, 3, 6), splitters = (5, 10, 30)</tt> defines the three intervals <tt>[-infinity,5), [5,10), [10,30)</tt>. Lets
   * define to sort the entire list (<tt>from=0, to=7</tt>) using all splitters (<tt>splitFrom==0, splitTo=2</tt>). <p>
   * The method modifies the list to be <tt>list = (4, 4, 3, 6, 7, 5, 6, 50)</tt> and returns the <tt>splitIndexes = (2,
   * 6, 6)</tt>. In other words, <ul> <li>All values <tt>list[0..2]</tt> fall into <tt>[-infinity,5)</tt>. <li>All
   * values <tt>list[3..6]</tt> fall into <tt>[5,10)</tt>. <li>All values <tt>list[7..6]</tt> fall into
   * <tt>[10,30)</tt>, i.e. no elements, since <tt>7>6</tt>. <li>All values <tt>list[7 .. 7=list.length-1]</tt> fall
   * into <tt>[30,infinity]</tt>. <li>In general, all values <tt>list[splitIndexes[j-1]+1 .. splitIndexes[j]]</tt> fall
   * into interval <tt>j</tt>. </ul> As can be seen, the list is partially sorted such that values falling into a
   * certain interval are placed next to each other. Note that <i>within</i> an interval, elements are entirelly
   * unsorted. They are only sorted across interval boundaries. In particular, this partitioning algorithm is not
   * <i>stable</i>: the relative order of elements is not preserved (Producing a stable algorithm would require no more
   * than minor modifications to method partition(int[],int,int,int)). <p> More formally, this method guarantees that
   * upon return <tt>for all j = splitFrom .. splitTo</tt> there holds: <br><tt>for all i = splitIndexes[j-1]+1 ..
   * splitIndexes[j]: splitters[j-1] <= list[i] < splitters[j]</tt>. <p> <b>Performance:</b> <p> Let
   * <tt>N=to-from+1</tt> be the number of elements to be partitioned. Let <tt>k=splitTo-splitFrom+1</tt> be the number
   * of splitter elements. Then we have the following time complexities <ul> <li>Worst case:  <tt>O( N * log(k) )</tt>.
   * <li>Average case: <tt>O( N * log(k) )</tt>. <li>Best case: <tt>O( N )</tt>. In general, the more uniform (skewed)
   * the data is spread across intervals, the more performance approaches the worst (best) case. If no elements fall
   * into the given intervals, running time is linear. </ul> No temporary memory is allocated; the sort is in-place. <p>
   * <b>Implementation:</b> <p> The algorithm can be seen as a Bentley/McIlroy quicksort where swapping and insertion
   * sort are omitted. It is designed to detect and take advantage of skew while maintaining good performance in the
   * uniform case.
   *
   * @param list         the list to be partially sorted.
   * @param from         the index of the first element within <tt>list</tt> to be considered.
   * @param to           the index of the last element within <tt>list</tt> to be considered. The method considers the
   *                     elements <tt>list[from] .. list[to]</tt>.
   * @param splitters    the values at which the list shall be split into intervals. Must be sorted ascending and must
   *                     not contain multiple identical values. These preconditions are not checked; be sure that they
   *                     are met.
   * @param splitFrom    the index of the first splitter element to be considered.
   * @param splitTo      the index of the last splitter element to be considered. The method considers the splitter
   *                     elements <tt>splitters[splitFrom] .. splitters[splitTo]</tt>.
   * @param splitIndexes a list into which this method fills the indexes of elements delimiting intervals. Upon return
   *                     <tt>splitIndexes[splitFrom..splitTo]</tt> will be set accordingly. Therefore, must satisfy
   *                     <tt>splitIndexes.length > splitTo</tt>. <p> Tip: Normally you will have <tt>splitIndexes.length
   *                     == splitters.length</tt> as well as <tt>from==0, to==list.length-1</tt> and <tt>splitFrom==0,
   *                     splitTo==splitters.length-1</tt>.
   * @see org.apache.mahout.math.Arrays
   * @see org.apache.mahout.math.GenericSorting
   * @see java.util.Arrays
   */
  public static void partition(int[] list, int from, int to, int[] splitters, int splitFrom, int splitTo,
                               int[] splitIndexes) {
    //int element;  // int, double --> template type dependent

    if (splitFrom > splitTo) {
      return;
    } // nothing to do
    if (from > to) { // all bins are empty
      from--;
      for (int i = splitFrom; i <= splitTo;) {
        splitIndexes[i++] = from;
      }
      return;
    }

    // Choose a partition (pivot) index, m
    // Ideally, the pivot should be the median, because a median splits a list into two equal sized sublists.
    // However, computing the median is expensive, so we use an approximation.
    int medianIndex;
    if (splitFrom == splitTo) { // we don't really have a choice
      medianIndex = splitFrom;
    } else { // we do have a choice
      int m = (from + to) / 2;       // Small arrays, middle element
      int len = to - from + 1;
      if (len > SMALL) {
        int l = from;
        int n = to;
        if (len > MEDIUM) {        // Big arrays, pseudomedian of 9
          int s = len / 8;
          l = med3(list, l, l + s, l + 2 * s);
          m = med3(list, m - s, m, m + s);
          n = med3(list, n - 2 * s, n - s, n);
        }
        m = med3(list, l, m, n); // Mid-size, pseudomedian of 3
      }

      // Find the splitter closest to the pivot, i.e. the splitter that best splits the list into two equal sized sublists.
      medianIndex = Sorting.binarySearchFromTo(splitters, list[m], splitFrom, splitTo);

      //int key = list[m];
      /*
      if (splitTo-splitFrom+1 < 5) { // on short lists linear search is quicker
        int i=splitFrom-1;
        while (++i <= splitTo && list[i] < key);
        if (i > splitTo || list[i] > key) i = -i-1; // not found
        medianIndex = i;
      }
      */
      //else {
      /*

        int low = splitFrom;
        int high = splitTo;
        int comparison;

        int mid=0;
        while (low <= high) {
          mid = (low + high) / 2;
          comparison = splitters[mid]-key;
          if (comparison < 0) low = mid + 1;
          else if (comparison > 0) high = mid - 1;
          else break; //return mid; // key found
        }
        medianIndex = mid;
        if (low > high) medianIndex = -(medianIndex + 1);  // key not found.
      //}
      */


      if (medianIndex < 0) {
        medianIndex = -medianIndex - 1;
      } // not found
      if (medianIndex > splitTo) {
        medianIndex = splitTo;
      } // not found, one past the end

    }
    int splitter = splitters[medianIndex];


    // Partition the list according to the splitter, i.e.
    // Establish invariant: list[i] < splitter <= list[j] for i=from..medianIndex and j=medianIndex+1 .. to
    // Could simply call:
    int splitIndex = partition(list, from, to, splitter);
    // but for speed the code is manually inlined.
    /*
    steps += to-from+1;
    int head = from;
    for (int i=from-1; ++i<=to; ) { // swap all elements < splitter to front
      element = list[i];
      if (element < splitter) {
        list[i] = list[head];
        list[head++] = element;
        //swappedElements++;
      }
    }
    int splitIndex = head-1;
    */



    splitIndexes[medianIndex] = splitIndex;

    //if (splitFrom == splitTo) return; // done

    // Optimization: Handle special cases to cut down recursions.
    if (splitIndex < from) { // no element falls into this bin
      // all bins with splitters[i] <= splitter are empty
      int i = medianIndex - 1;
      while (i >= splitFrom && (!(splitter < splitters[i]))) {
        splitIndexes[i--] = splitIndex;
      }
      splitFrom = medianIndex + 1;
    } else if (splitIndex >= to) { // all elements fall into this bin
      // all bins with splitters[i] >= splitter are empty
      int i = medianIndex + 1;
      while (i <= splitTo && (!(splitter > splitters[i]))) {
        splitIndexes[i++] = splitIndex;
      }
      splitTo = medianIndex - 1;
    }

    // recursively partition left half
    if (splitFrom <= medianIndex - 1) {
      partition(list, from, splitIndex, splitters, splitFrom, medianIndex - 1, splitIndexes);
    }

    // recursively partition right half
    if (medianIndex + 1 <= splitTo) {
      partition(list, splitIndex + 1, to, splitters, medianIndex + 1, splitTo, splitIndexes);
    }
    //log.info("BACK TRACKING\n\n");
  }

  /**
   * Partitions (partially sorts) the given list such that all elements falling into the given interval are placed next
   * to each other. Returns the index of the element delimiting the interval. <p> <b>Example:</b> <p> <tt>list = (7, 4,
   * 5, 50, 6, 4, 3, 6), splitter = 5</tt> defines the two intervals <tt>[-infinity,5), [5,+infinity]</tt>. <p> The
   * method modifies the list to be <tt>list = (4, 4, 3, 50, 6, 7, 5, 6)</tt> and returns the split index <tt>2</tt>. In
   * other words, <ul> <li>All values <tt>list[0..2]</tt> fall into <tt>[-infinity,5)</tt>. <li>All values
   * <tt>list[3=2+1 .. 7=list.length-1]</tt> fall into <tt>[5,+infinity]</tt>. </ul> As can be seen, the list is
   * partially sorted such that values falling into a certain interval are placed next to each other. Note that
   * <i>within</i> an interval, elements are entirelly unsorted. They are only sorted across interval boundaries. In
   * particular, this partitioning algorithm is not <i>stable</i>. <p> More formally, this method guarantees that upon
   * return there holds: <ul> <li>for all <tt>i = from .. returnValue: list[i] < splitter</tt> and <li>for all <tt>i =
   * returnValue+1 .. list.length-1: !(list[i] < splitter)</tt>. </ul> <p> <b>Performance:</b> <p> Let
   * <tt>N=to-from+1</tt> be the number of elements to be partially sorted. Then the time complexity is <tt>O( N )</tt>.
   * No temporary memory is allocated; the sort is in-place.
   *
   * <p>
   *
   * @param list     the list to be partially sorted.
   * @param from     the index of the first element within <tt>list</tt> to be considered.
   * @param to       the index of the last element within <tt>list</tt> to be considered. The method considers the
   *                 elements <tt>list[from] .. list[to]</tt>.
   * @param splitter the value at which the list shall be split.
   * @return the index of the largest element falling into the interval <tt>[-infinity,splitter)</tt>, as seen after
   *         partitioning.
   */
  public static int partition(int[] list, int from, int to, int splitter) {
    steps += to - from + 1;

    for (int i = from - 1; ++i <= to;) {
      int element = list[i];
      if (element < splitter) {
        // swap x[i] with x[from]
        list[i] = list[from];
        list[from++] = element;
        //swappedElements++;
      }
    }

    return from - 1;
  }

  /**
   * Same as {@link #partition(int[],int,int,int[],int,int,int[])} except that it partitions <tt>Object[]</tt> rather
   * than <tt>int[]</tt> arrays.
   */
  public static void partition(Object[] list, int from, int to, Object[] splitters, int splitFrom, int splitTo,
                               int[] splitIndexes, Comparator<Object> comp) {

    if (splitFrom > splitTo) {
      return;
    } // nothing to do
    if (from > to) { // all bins are empty
      from--;
      for (int i = splitFrom; i <= splitTo;) {
        splitIndexes[i++] = from;
      }
      return;
    }

    // Choose a partition (pivot) index, m
    // Ideally, the pivot should be the median, because a median splits a list into two equal sized sublists.
    // However, computing the median is expensive, so we use an approximation.
    int medianIndex;
    if (splitFrom == splitTo) { // we don't really have a choice
      medianIndex = splitFrom;
    } else { // we do have a choice
      int m = (from + to) / 2;       // Small arrays, middle element
      int len = to - from + 1;
      if (len > SMALL) {
        int l = from;
        int n = to;
        if (len > MEDIUM) {        // Big arrays, pseudomedian of 9
          int s = len / 8;
          l = med3(list, l, l + s, l + 2 * s, comp);
          m = med3(list, m - s, m, m + s, comp);
          n = med3(list, n - 2 * s, n - s, n, comp);
        }
        m = med3(list, l, m, n, comp); // Mid-size, pseudomedian of 3
      }

      // Find the splitter closest to the pivot, i.e. the splitter that best splits the list into two equal sized sublists.
      medianIndex = Sorting.binarySearchFromTo(splitters, list[m], splitFrom, splitTo, comp);
      if (medianIndex < 0) {
        medianIndex = -medianIndex - 1;
      } // not found
      if (medianIndex > splitTo) {
        medianIndex = splitTo;
      } // not found, one past the end

    }
    Object splitter = splitters[medianIndex]; // int, double --> template type dependent

    // Partition the list according to the splitter, i.e.
    // Establish invariant: list[i] < splitter <= list[j] for i=from..medianIndex and j=medianIndex+1 .. to
    int splitIndex = partition(list, from, to, splitter, comp);
    splitIndexes[medianIndex] = splitIndex;

    // Optimization: Handle special cases to cut down recursions.
    if (splitIndex < from) { // no element falls into this bin
      // all bins with splitters[i] <= splitter are empty
      int i = medianIndex - 1;
      while (i >= splitFrom && (!(comp.compare(splitter, splitters[i]) < 0))) {
        splitIndexes[i--] = splitIndex;
      }
      splitFrom = medianIndex + 1;
    } else if (splitIndex >= to) { // all elements fall into this bin
      // all bins with splitters[i] >= splitter are empty
      int i = medianIndex + 1;
      while (i <= splitTo && (!(comp.compare(splitter, splitters[i]) > 0))) {
        splitIndexes[i++] = splitIndex;
      }
      splitTo = medianIndex - 1;
    }

    // recursively partition left half
    if (splitFrom <= medianIndex - 1) {
      partition(list, from, splitIndex, splitters, splitFrom, medianIndex - 1, splitIndexes, comp);
    }

    // recursively partition right half
    if (medianIndex + 1 <= splitTo) {
      partition(list, splitIndex + 1, to, splitters, medianIndex + 1, splitTo, splitIndexes, comp);
    }
  }

  /**
   * Same as {@link #partition(int[],int,int,int)} except that it <i>synchronously</i> partitions the objects of the
   * given list by the order of the given comparator.
   */
  public static int partition(Object[] list, int from, int to, Object splitter, Comparator<Object> comp) {
    for (int i = from - 1; ++i <= to;) {
      Object element = list[i];  // int, double --> template type dependent
      if (comp.compare(element, splitter) < 0) {
        // swap x[i] with x[from]
        list[i] = list[from];
        list[from] = element;
        from++;
      }
    }
    return from - 1;
  }

  /**
   * Equivalent to <tt>partition(list.elements(), from, to, splitters.elements(), 0, splitters.size()-1,
   * splitIndexes.elements())</tt>.
   */
  public static void partition(DoubleArrayList list, int from, int to, DoubleArrayList splitters,
                               IntArrayList splitIndexes) {
    partition(list.elements(), from, to, splitters.elements(), 0, splitters.size() - 1, splitIndexes.elements());
  }

  /**
   * Equivalent to <tt>partition(list.elements(), from, to, splitters.elements(), 0, splitters.size()-1,
   * splitIndexes.elements())</tt>.
   */
  public static void partition(IntArrayList list, int from, int to, IntArrayList splitters, IntArrayList splitIndexes) {
    partition(list.elements(), from, to, splitters.elements(), 0, splitters.size() - 1, splitIndexes.elements());
  }

  /**
   * Same as {@link #triplePartition(int[],int[],int[],int,int,int[],int,int,int[])} except that it <i>synchronously</i>
   * partitions <tt>double[]</tt> rather than <tt>int[]</tt> arrays.
   */
  public static void triplePartition(double[] list, double[] secondary, double[] tertiary, int from, int to,
                                     double[] splitters, int splitFrom, int splitTo, int[] splitIndexes) {

    if (splitFrom > splitTo) {
      return;
    } // nothing to do
    if (from > to) { // all bins are empty
      from--;
      for (int i = splitFrom; i <= splitTo;) {
        splitIndexes[i++] = from;
      }
      return;
    }

    // Choose a partition (pivot) index, m
    // Ideally, the pivot should be the median, because a median splits a list into two equal sized sublists.
    // However, computing the median is expensive, so we use an approximation.
    int medianIndex;
    if (splitFrom == splitTo) { // we don't really have a choice
      medianIndex = splitFrom;
    } else { // we do have a choice
      int m = (from + to) / 2;       // Small arrays, middle element
      int len = to - from + 1;
      if (len > SMALL) {
        int l = from;
        int n = to;
        if (len > MEDIUM) {        // Big arrays, pseudomedian of 9
          int s = len / 8;
          l = med3(list, l, l + s, l + 2 * s);
          m = med3(list, m - s, m, m + s);
          n = med3(list, n - 2 * s, n - s, n);
        }
        m = med3(list, l, m, n); // Mid-size, pseudomedian of 3
      }

      // Find the splitter closest to the pivot, i.e. the splitter that best splits the list into two equal sized sublists.
      medianIndex = Sorting.binarySearchFromTo(splitters, list[m], splitFrom, splitTo);
      if (medianIndex < 0) {
        medianIndex = -medianIndex - 1;
      } // not found
      if (medianIndex > splitTo) {
        medianIndex = splitTo;
      } // not found, one past the end

    }
    double splitter = splitters[medianIndex]; // int, double --> template type dependent

    // Partition the list according to the splitter, i.e.
    // Establish invariant: list[i] < splitter <= list[j] for i=from..medianIndex and j=medianIndex+1 .. to
    int splitIndex = triplePartition(list, secondary, tertiary, from, to, splitter);
    splitIndexes[medianIndex] = splitIndex;

    // Optimization: Handle special cases to cut down recursions.
    if (splitIndex < from) { // no element falls into this bin
      // all bins with splitters[i] <= splitter are empty
      int i = medianIndex - 1;
      while (i >= splitFrom && (!(splitter < splitters[i]))) {
        splitIndexes[i--] = splitIndex;
      }
      splitFrom = medianIndex + 1;
    } else if (splitIndex >= to) { // all elements fall into this bin
      // all bins with splitters[i] >= splitter are empty
      int i = medianIndex + 1;
      while (i <= splitTo && (!(splitter > splitters[i]))) {
        splitIndexes[i++] = splitIndex;
      }
      splitTo = medianIndex - 1;
    }

    // recursively partition left half
    if (splitFrom <= medianIndex - 1) {
      triplePartition(list, secondary, tertiary, from, splitIndex, splitters, splitFrom, medianIndex - 1, splitIndexes);
    }

    // recursively partition right half
    if (medianIndex + 1 <= splitTo) {
      triplePartition(list, secondary, tertiary, splitIndex + 1, to, splitters, medianIndex + 1, splitTo, splitIndexes);
    }
  }

  /**
   * Same as {@link #triplePartition(int[],int[],int[],int,int,int)} except that it <i>synchronously</i> partitions
   * <tt>double[]</tt> rather than <tt>int[]</tt> arrays.
   */
  public static int triplePartition(double[] list, double[] secondary, double[] tertiary, int from, int to,
                                    double splitter) {
    for (int i = from - 1; ++i <= to;) {
      double element = list[i];  // int, double --> template type dependent
      if (element < splitter) {
        // swap x[i] with x[from]
        list[i] = list[from];
        list[from] = element;

        element = secondary[i];
        secondary[i] = secondary[from];
        secondary[from] = element;

        element = tertiary[i];
        tertiary[i] = tertiary[from];
        tertiary[from++] = element;
      }
    }

    return from - 1;
  }

  /**
   * Same as {@link #partition(int[],int,int,int[],int,int,int[])} except that this method <i>synchronously</i>
   * partitions three arrays at the same time; all three arrays are partially sorted according to the elements of the
   * primary array. In other words, each time an element in the primary array is moved from index A to B, the
   * correspoding element within the secondary array as well as the corresponding element within the tertiary array are
   * also moved from index A to B. <p> <b>Use cases:</b> <p> Image having a large list of 3-dimensional points. If
   * memory consumption and performance matter, it is a good idea to physically lay them out as three 1-dimensional
   * arrays (using something like <tt>Point3D</tt> objects would be prohibitively expensive, both in terms of time and
   * space). Now imagine wanting to histogram the points. We may want to partially sort the points by x-coordinate into
   * intervals. This method efficiently does the job. <p> <b>Performance:</b> <p> Same as for single-partition methods.
   */
  public static void triplePartition(int[] list, int[] secondary, int[] tertiary, int from, int to, int[] splitters,
                                     int splitFrom, int splitTo, int[] splitIndexes) {

    if (splitFrom > splitTo) {
      return;
    } // nothing to do
    if (from > to) { // all bins are empty
      from--;
      for (int i = splitFrom; i <= splitTo;) {
        splitIndexes[i++] = from;
      }
      return;
    }

    // Choose a partition (pivot) index, m
    // Ideally, the pivot should be the median, because a median splits a list into two equal sized sublists.
    // However, computing the median is expensive, so we use an approximation.
    int medianIndex;
    if (splitFrom == splitTo) { // we don't really have a choice
      medianIndex = splitFrom;
    } else { // we do have a choice
      int m = (from + to) / 2;       // Small arrays, middle element
      int len = to - from + 1;
      if (len > SMALL) {
        int l = from;
        int n = to;
        if (len > MEDIUM) {        // Big arrays, pseudomedian of 9
          int s = len / 8;
          l = med3(list, l, l + s, l + 2 * s);
          m = med3(list, m - s, m, m + s);
          n = med3(list, n - 2 * s, n - s, n);
        }
        m = med3(list, l, m, n); // Mid-size, pseudomedian of 3
      }

      // Find the splitter closest to the pivot, i.e. the splitter that best splits the list into two equal sized sublists.
      medianIndex = Sorting.binarySearchFromTo(splitters, list[m], splitFrom, splitTo);
      if (medianIndex < 0) {
        medianIndex = -medianIndex - 1;
      } // not found
      if (medianIndex > splitTo) {
        medianIndex = splitTo;
      } // not found, one past the end

    }
    int splitter = splitters[medianIndex]; // int, double --> template type dependent

    // Partition the list according to the splitter, i.e.
    // Establish invariant: list[i] < splitter <= list[j] for i=from..medianIndex and j=medianIndex+1 .. to
    int splitIndex = triplePartition(list, secondary, tertiary, from, to, splitter);
    splitIndexes[medianIndex] = splitIndex;

    // Optimization: Handle special cases to cut down recursions.
    if (splitIndex < from) { // no element falls into this bin
      // all bins with splitters[i] <= splitter are empty
      int i = medianIndex - 1;
      while (i >= splitFrom && (!(splitter < splitters[i]))) {
        splitIndexes[i--] = splitIndex;
      }
      splitFrom = medianIndex + 1;
    } else if (splitIndex >= to) { // all elements fall into this bin
      // all bins with splitters[i] >= splitter are empty
      int i = medianIndex + 1;
      while (i <= splitTo && (!(splitter > splitters[i]))) {
        splitIndexes[i++] = splitIndex;
      }
      splitTo = medianIndex - 1;
    }

    // recursively partition left half
    if (splitFrom <= medianIndex - 1) {
      triplePartition(list, secondary, tertiary, from, splitIndex, splitters, splitFrom, medianIndex - 1, splitIndexes);
    }

    // recursively partition right half
    if (medianIndex + 1 <= splitTo) {
      triplePartition(list, secondary, tertiary, splitIndex + 1, to, splitters, medianIndex + 1, splitTo, splitIndexes);
    }
  }

  /**
   * Same as {@link #partition(int[],int,int,int)} except that this method <i>synchronously</i> partitions three arrays
   * at the same time; all three arrays are partially sorted according to the elements of the primary array. In other
   * words, each time an element in the primary array is moved from index A to B, the correspoding element within the
   * secondary array as well as the corresponding element within the tertiary array are also moved from index A to B.
   * <p> <b>Performance:</b> <p> Same as for single-partition methods.
   */
  public static int triplePartition(int[] list, int[] secondary, int[] tertiary, int from, int to, int splitter) {
    for (int i = from - 1; ++i <= to;) {
      int element = list[i];  // int, double --> template type dependent
      if (element < splitter) {
        // swap x[i] with x[from]
        list[i] = list[from];
        list[from] = element;

        element = secondary[i];
        secondary[i] = secondary[from];
        secondary[from] = element;

        element = tertiary[i];
        tertiary[i] = tertiary[from];
        tertiary[from++] = element;
      }
    }

    return from - 1;
  }
}
