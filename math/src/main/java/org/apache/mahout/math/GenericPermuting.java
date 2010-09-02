/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.jet.random.Uniform;
import org.apache.mahout.math.jet.random.engine.MersenneTwister;

/**
 Generically reorders (permutes) arbitrary shaped data (for example, an array, three arrays, a 2-d matrix, two linked lists) using an <i>in-place</i> swapping algorithm.
 Imagine having a couple of apples. For some reason you decide to reorder them. The green one before the red one. The pale one after the shiny one, etc. This class helps to do the job.
 <p>
 This class swaps elements around, in a way that avoids stumbling over its own feet:
 Let <tt>before</tt> be the generic data before calling the reordering method.
 Let <tt>after</tt> be the generic data after calling the reordering method.
 Then there holds <tt>after[i] == before[indexes[i]]</tt>.
 <p>
 Similar to {@link GenericSorting}, this class has no idea what kind of data it is reordering.
 It can decide to swap the data at index <tt>a</tt> with the data at index <tt>b</tt>.
 It calls a user provided {@link org.apache.mahout.math.Swapper} object that knows how to swap the data of these indexes.
 <p>
 For convenience, some non-generic variants are also provided.
 Further a method to generate the p-th lexicographical permutation indexes.
 <p>
 <b>Example:</b>
 <table>
 <td class="PRE">
 <pre>
 Reordering
 [A,B,C,D,E] with indexes [0,4,2,3,1] yields
 [A,E,C,D,B]
 In other words, in the reordered list, we first have the element at old index 0, then the one at old index 4, then the ones at old indexes 2,3,1.
 g[0]<--g[0], g[1]<--g[4], g[2]<--g[2], g[3]<--g[3], g[4]<--g[1].

 Reordering
 [A,B,C,D,E] with indexes [0,4,1,2,3] yields
 [A,E,B,C,D]
 In other words g[0]<--g[0], g[1]<--g[4], g[2]<--g[1], g[3]<--g[2], g[4]<--g[3].
 </pre>
 </td>
 </table>
 <p>
 Here are some example swappers:
 <table>
 <td class="PRE">
 <pre>
 // a swapper knows how to swap two indexes (a,b)

 // reordering an array
 Swapper swapper = new Swapper() {
 &nbsp;&nbsp;&nbsp;public void swap(int a, int b) {
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;String tmp; // reordering String[]
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;// int tmp; // reordering int[]
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tmp = array[a]; array[a] = array[b]; array[b] = tmp;
 &nbsp;&nbsp;&nbsp;}
 };

 // reordering a list
 Swapper swapper = new Swapper() {
 &nbsp;&nbsp;&nbsp;public void swap(int a, int b) {
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Object tmp;
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;tmp = list.get(a);
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;list.set(a, list.get(b));
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;list.set(b, tmp);
 &nbsp;&nbsp;&nbsp;}
 };

 // reordering the rows of a 2-d matrix (see {@link org.apache.mahout.math.matrix})
 Swapper swapper = new Swapper() {
 &nbsp;&nbsp;&nbsp;public void swap(int a, int b) {
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;matrix.viewRow(a).swap(matrix.viewRow(b));
 &nbsp;&nbsp;&nbsp;}
 };

 // reordering the columns of a 2-d matrix
 Swapper swapper = new Swapper() {
 &nbsp;&nbsp;&nbsp;public void swap(int a, int b) {
 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;matrix.viewColumn(a).swap(matrix.viewColumn(b));
 &nbsp;&nbsp;&nbsp;}
 };
 </pre>
 </td>
 </table>

 @see org.apache.mahout.math.Swapper
 @see org.apache.mahout.math.GenericSorting

 @author wolfgang.hoschek@cern.ch
 @version 1.0, 10-Oct-99
 */

/** @deprecated until unit tests are in place.  Until this time, this class/interface is unsupported. */
@Deprecated
public class GenericPermuting {

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private GenericPermuting() {
  }

  /**
   * Returns the <tt>p</tt>-th permutation of the sequence <tt>[0,1,...,N-1]</tt>. A small but smart and efficient
   * routine, ported from <A HREF="http://www.hep.net/wwwmirrors/cernlib/CNASDOC/shortwrups_html3/node255.html">
   * Cernlib</A>. The <A HREF="ftp://asisftp.cern.ch/cernlib/share/pro/src/mathlib/gen/v/permu.F"> Fortran source</A>. A
   * sequence of <tt>N</tt> distinct elements has <tt>N!</tt> permutations, which are enumerated in lexicographical
   * order <tt>1 .. N!</tt>. <p> This is, for example, useful for Monte-Carlo-tests where one might want to compute
   * <tt>k</tt> distinct and random permutations of a sequence, obtaining <tt>p</tt> from {@link
   * org.apache.mahout.math.jet.random.sampling} without replacement or a random engine like {@link
   * org.apache.mahout.math.jet.random.engine.MersenneTwister}. <br> Note: When <tt>N!</tt> exceeds the 64-bit range (i.e.
   * for <tt>N > 20</tt>), this method has <i>different</i> behaviour: it makes a sequence <tt>[0,1,...,N-1]</tt> and
   * randomizes it, seeded with parameter <tt>p</tt>. <p> <b>Examples:</b>
   * <pre>
   * http://www.hep.net/wwwmirrors/cernlib/CNASDOC/shortwrups_html3/node255.html
   * // exactly lexicographically enumerated (ascending)
   * permutation(1,3) --> [ 0,1,2 ]
   * permutation(2,3) --> [ 0,2,1 ]
   * permutation(3,3) --> [ 1,0,2 ]
   * permutation(4,3) --> [ 1,2,0 ]
   * permutation(5,3) --> [ 2,0,1 ]
   * permutation(6,3) --> [ 2,1,0 ]
   * permutation(1      ,20) --> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
   * permutation(2      ,20) --> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 18]
   * permutation(1000000,20) --> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 17, 18, 13, 19, 11, 15, 14, 16, 10]
   * permutation(20! -2 ,20) --> [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 1, 2, 0]
   * permutation(20! -1 ,20) --> [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 0, 1]
   * permutation(20!    ,20) --> [19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
   * <br>
   * // not exactly enumerated, rather randomly shuffled
   * permutation(1,21) --> [18, 20, 11, 0, 15, 1, 19, 13, 3, 6, 16, 17, 9, 5, 12, 4, 7, 14, 8, 10, 2]
   * permutation(2,21) --> [1, 9, 4, 16, 14, 13, 11, 20, 10, 8, 18, 0, 15, 3, 17, 5, 12, 2, 6, 7, 19]
   * permutation(3,21) --> [12, 0, 19, 1, 20, 5, 8, 16, 6, 14, 2, 4, 3, 17, 11, 13, 9, 10, 15, 18, 7]
   * </pre>
   *
   * @param p the lexicographical ordinal number of the permutation to be computed.
   * @param N the length of the sequence to be generated.
   * @return the <tt>p</tt>-th permutation.
   * @throws IllegalArgumentException if <tt>p < 1 || N < 0 || p > N!</tt>.
   */
  public static int[] permutation(long p, int N) {
    if (p < 1) {
      throw new IllegalArgumentException("Permutations are enumerated 1 .. N!");
    }
    if (N < 0) {
      throw new IllegalArgumentException("Must satisfy N >= 0");
    }

    int[] permutation = new int[N];

    if (N > 20) { // factorial(21) would overflow 64-bit long)
      // Simply make a list (0,1,..N-1) and randomize it, seeded with "p".
      // Note that this is perhaps not what you want...
      for (int i = N; --i >= 0;) {
        permutation[i] = i;
      }
      Uniform gen = new Uniform(RandomUtils.getRandom());
      for (int i = 0; i < N - 1; i++) {
        int random = gen.nextIntFromTo(i, N - 1);

        //swap(i, random)
        int tmp = permutation[random];
        permutation[random] = permutation[i];
        permutation[i] = tmp;
      }

      return permutation;
    }

    // the normal case - exact enumeration
    if (p > org.apache.mahout.math.jet.math.Arithmetic.longFactorial(N)) {
      throw new IllegalArgumentException("N too large (a sequence of N elements only has N! permutations).");
    }

    int[] tmp = new int[N];
    for (int i = 1; i <= N; i++) {
      tmp[i - 1] = i;
    }

    long io = p - 1;
    for (int M = N - 1; M >= 1; M--) {
      long fac = org.apache.mahout.math.jet.math.Arithmetic.longFactorial(M);
      int in = ((int) (io / fac)) + 1;
      io %= fac;
      permutation[N - M - 1] = tmp[in - 1];

      for (int j = in; j <= M; j++) {
        tmp[j - 1] = tmp[j];
      }
    }
    if (N > 0) {
      permutation[N - 1] = tmp[0];
    }

    for (int i = N; --i >= 0;) {
      permutation[i] -= 1;
    }
    return permutation;
  }

  /**
   * A non-generic variant of reordering, specialized for <tt>int[]</tt>, same semantics. Quicker than generic
   * reordering. Also for convenience (forget about the Swapper object).
   */
  public static void permute(int[] list, int[] indexes) {
    int[] copy = list.clone();
    for (int i = list.length; --i >= 0;) {
      list[i] = copy[indexes[i]];
    }
  }

  /**
   * Deprecated. Generically reorders arbitrary shaped generic data <tt>g</tt> such that <tt>g[i] == g[indexes[i]]</tt>.
   * (The generic data may be one array, a 2-d matrix, two linked lists or whatever). This class swaps elements around,
   * in a way that avoids stumbling over its own feet. <p> <b>Example:</b>
   * <pre>
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,2,3,1] yields
   * [A,E,C,D,B]
   * In other words g[0]<--g[0], g[1]<--g[4], g[2]<--g[2], g[3]<--g[3], g[4]<--g[1].
   *
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,1,2,3] yields
   * [A,E,B,C,D]
   * In other words g[0]<--g[0], g[1]<--g[4], g[2]<--g[1], g[3]<--g[2], g[4]<--g[3].
   * </pre>
   * <p>
   *
   * @param indexes the permutation indexes.
   * @param swapper an object that knows how to swap two indexes a,b.
   * @param work    the working storage, must satisfy <tt>work.length >= indexes.length</tt>; set <tt>work==null</tt> if
   *                you don't care about performance.
   * @deprecated
   */
  @Deprecated
  public static void permute(int[] indexes, org.apache.mahout.math.Swapper swapper, int[] work) {
    permute(indexes, swapper, work, null);
  }

  /**
   * Generically reorders arbitrary shaped generic data <tt>g</tt> such that <tt>g[i] == g[indexes[i]]</tt>. (The
   * generic data may be one array, a 2-d matrix, two linked lists or whatever). This class swaps elements around, in a
   * way that avoids stumbling over its own feet. <p> <b>Example:</b>
   * <pre>
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,2,3,1] yields
   * [A,E,C,D,B]
   * In other words g[0]<--g[0], g[1]<--g[4], g[2]<--g[2], g[3]<--g[3], g[4]<--g[1].
   *
   * Reordering
   * [A,B,C,D,E] with indexes [0,4,1,2,3] yields
   * [A,E,B,C,D]
   * In other words g[0]<--g[0], g[1]<--g[4], g[2]<--g[1], g[3]<--g[2], g[4]<--g[3].
   * </pre>
   * <p>
   *
   * @param indexes the permutation indexes.
   * @param swapper an object that knows how to swap two indexes a,b.
   * @param work1   some working storage, must satisfy <tt>work1.length >= indexes.length</tt>; set <tt>work1==null</tt>
   *                if you don't care about performance.
   * @param work2   some working storage, must satisfy <tt>work2.length >= indexes.length</tt>; set <tt>work2==null</tt>
   *                if you don't care about performance.
   */
  public static void permute(int[] indexes, org.apache.mahout.math.Swapper swapper, int[] work1, int[] work2) {
    // "tracks" and "pos" keeps track of the current indexes of the elements
    // Example: We have a list==[A,B,C,D,E], indexes==[0,4,1,2,3] and swap B and E we need to know that the element formlerly at index 1 is now at index 4, and the one formerly at index 4 is now at index 1.
    // Otherwise we stumble over our own feet and produce nonsense.
    // Initially index i really is at index i, but this will change due to swapping.

    // work1, work2 to avoid high frequency memalloc's
    int s = indexes.length;
    int[] tracks = work1;
    int[] pos = work2;
    if (tracks == null || tracks.length < s) {
      tracks = new int[s];
    }
    if (pos == null || pos.length < s) {
      pos = new int[s];
    }
    for (int i = s; --i >= 0;) {
      tracks[i] = i;
      pos[i] = i;
    }

    for (int i = 0; i < s; i++) {
      int index = indexes[i];
      int track = tracks[index];

      if (i != track) {
        swapper.swap(i, track);
        tracks[index] = i;
        tracks[pos[i]] = track;
        int tmp = pos[i];
        pos[i] = pos[track];
        pos[track] = tmp;
      }
    }
  }

  /**
   * A non-generic variant of reordering, specialized for <tt>Object[]</tt>, same semantics. Quicker than generic
   * reordering. Also for convenience (forget about the Swapper object).
   */
  public static void permute(Object[] list, int[] indexes) {
    Object[] copy = list.clone();
    for (int i = list.length; --i >= 0;) {
      list[i] = copy[indexes[i]];
    }
  }
}
