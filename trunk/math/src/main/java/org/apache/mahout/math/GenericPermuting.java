/*
Copyright � 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose 
is hereby granted without fee, provided that the above copyright notice appear in all copies and 
that both that copyright notice and this permission notice appear in supporting documentation. 
CERN makes no representations about the suitability of this software for any purpose. 
It is provided "as is" without expressed or implied warranty.
*/
package org.apache.mahout.math;

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
public final class GenericPermuting {

  /** Makes this class non instantiable, but still let's others inherit from it. */
  private GenericPermuting() {
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

}
