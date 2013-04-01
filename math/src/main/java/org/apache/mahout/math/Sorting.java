/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math;

import java.io.Serializable;
import java.util.Comparator;

import com.google.common.base.Preconditions;
import org.apache.mahout.math.function.ByteComparator;
import org.apache.mahout.math.function.CharComparator;
import org.apache.mahout.math.function.DoubleComparator;
import org.apache.mahout.math.function.FloatComparator;
import org.apache.mahout.math.function.IntComparator;
import org.apache.mahout.math.function.LongComparator;
import org.apache.mahout.math.function.ShortComparator;

public final class Sorting {
  
  /* Specifies when to switch to insertion sort */
  private static final int SIMPLE_LENGTH = 7;
  static final int SMALL = 7;
  
  private Sorting() {}
  
  private static <T> int med3(T[] array, int a, int b, int c, Comparator<T> comp) {
    T x = array[a];
    T y = array[b];
    T z = array[c];
    int comparisonxy = comp.compare(x, y);
    int comparisonxz = comp.compare(x, z);
    int comparisonyz = comp.compare(y, z);
    return comparisonxy < 0 ? (comparisonyz < 0 ? b
        : (comparisonxz < 0 ? c : a)) : (comparisonyz > 0 ? b
        : (comparisonxz > 0 ? c : a));
  }
  
  private static int med3(byte[] array, int a, int b, int c, ByteComparator comp) {
    byte x = array[a];
    byte y = array[b];
    byte z = array[c];
    int comparisonxy = comp.compare(x, y);
    int comparisonxz = comp.compare(x, z);
    int comparisonyz = comp.compare(y, z);
    return comparisonxy < 0 ? (comparisonyz < 0 ? b
        : (comparisonxz < 0 ? c : a)) : (comparisonyz > 0 ? b
        : (comparisonxz > 0 ? c : a));
  }
  
  private static int med3(char[] array, int a, int b, int c, CharComparator comp) {
    char x = array[a];
    char y = array[b];
    char z = array[c];
    int comparisonxy = comp.compare(x, y);
    int comparisonxz = comp.compare(x, z);
    int comparisonyz = comp.compare(y, z);
    return comparisonxy < 0 ? (comparisonyz < 0 ? b
        : (comparisonxz < 0 ? c : a)) : (comparisonyz > 0 ? b
        : (comparisonxz > 0 ? c : a));
  }
  
  private static int med3(double[] array, int a, int b, int c,
      DoubleComparator comp) {
    double x = array[a];
    double y = array[b];
    double z = array[c];
    int comparisonxy = comp.compare(x, y);
    int comparisonxz = comp.compare(x, z);
    int comparisonyz = comp.compare(y, z);
    return comparisonxy < 0 ? (comparisonyz < 0 ? b
        : (comparisonxz < 0 ? c : a)) : (comparisonyz > 0 ? b
        : (comparisonxz > 0 ? c : a));
  }
  
  private static int med3(float[] array, int a, int b, int c,
      FloatComparator comp) {
    float x = array[a];
    float y = array[b];
    float z = array[c];
    int comparisonxy = comp.compare(x, y);
    int comparisonxz = comp.compare(x, z);
    int comparisonyz = comp.compare(y, z);
    return comparisonxy < 0 ? (comparisonyz < 0 ? b
        : (comparisonxz < 0 ? c : a)) : (comparisonyz > 0 ? b
        : (comparisonxz > 0 ? c : a));
  }
  
  private static int med3(int[] array, int a, int b, int c, IntComparator comp) {
    int x = array[a];
    int y = array[b];
    int z = array[c];
    int comparisonxy = comp.compare(x, y);
    int comparisonxz = comp.compare(x, z);
    int comparisonyz = comp.compare(y, z);
    return comparisonxy < 0 ? (comparisonyz < 0 ? b
        : (comparisonxz < 0 ? c : a)) : (comparisonyz > 0 ? b
        : (comparisonxz > 0 ? c : a));
  }
  
  /**
   * This is used for 'external' sorting. The comparator takes <em>indices</em>,
   * not values, and compares the external values found at those indices.
   * @param a
   * @param b
   * @param c
   * @param comp
   * @return
   */
  private static int med3(int a, int b, int c, IntComparator comp) {
    int comparisonab = comp.compare(a, b);
    int comparisonac = comp.compare(a, c);
    int comparisonbc = comp.compare(b, c);
    return comparisonab < 0
        ? (comparisonbc < 0 ? b : (comparisonac < 0 ? c : a))
        : (comparisonbc > 0 ? b : (comparisonac > 0 ? c : a));
  }
  
  private static int med3(long[] array, int a, int b, int c, LongComparator comp) {
    long x = array[a];
    long y = array[b];
    long z = array[c];
    int comparisonxy = comp.compare(x, y);
    int comparisonxz = comp.compare(x, z);
    int comparisonyz = comp.compare(y, z);
    return comparisonxy < 0 ? (comparisonyz < 0 ? b
        : (comparisonxz < 0 ? c : a)) : (comparisonyz > 0 ? b
        : (comparisonxz > 0 ? c : a));
  }
  
  private static int med3(short[] array, int a, int b, int c,
      ShortComparator comp) {
    short x = array[a];
    short y = array[b];
    short z = array[c];
    int comparisonxy = comp.compare(x, y);
    int comparisonxz = comp.compare(x, z);
    int comparisonyz = comp.compare(y, z);
    return comparisonxy < 0 ? (comparisonyz < 0 ? b
        : (comparisonxz < 0 ? c : a)) : (comparisonyz > 0 ? b
        : (comparisonxz > 0 ? c : a));
  }
  
  /**
   * Sorts the specified range in the array in a specified order.
   * 
   * @param array
   *          the {@code byte} array to be sorted.
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @param comp
   *          the comparison that determines the sort.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   */
  public static void quickSort(byte[] array, int start, int end,
      ByteComparator comp) {
    Preconditions.checkNotNull(array);
    checkBounds(array.length, start, end);
    quickSort0(start, end, array, comp);
  }
  
  private static void checkBounds(int arrLength, int start, int end) {
    if (start > end) {
      // K0033=Start index ({0}) is greater than end index ({1})
      throw new IllegalArgumentException("Start index " + start
          + " is greater than end index " + end);
    }
    if (start < 0) {
      throw new ArrayIndexOutOfBoundsException("Array index out of range "
          + start);
    }
    if (end > arrLength) {
      throw new ArrayIndexOutOfBoundsException("Array index out of range "
          + end);
    }
  }
  
  private static void quickSort0(int start, int end, byte[] array, ByteComparator comp) {
    byte temp;
    int length = end - start;
    if (length < 7) {
      for (int i = start + 1; i < end; i++) {
        for (int j = i; j > start && comp.compare(array[j - 1], array[j]) > 0; j--) {
          temp = array[j];
          array[j] = array[j - 1];
          array[j - 1] = temp;
        }
      }
      return;
    }
    int middle = (start + end) / 2;
    if (length > 7) {
      int bottom = start;
      int top = end - 1;
      if (length > 40) {
        length /= 8;
        bottom = med3(array, bottom, bottom + length, bottom + (2 * length),
            comp);
        middle = med3(array, middle - length, middle, middle + length, comp);
        top = med3(array, top - (2 * length), top - length, top, comp);
      }
      middle = med3(array, bottom, middle, top, comp);
    }
    byte partionValue = array[middle];
    int a = start;
    int b = a;
    int c = end - 1;
    int d = c;
    while (true) {
      int comparison;
      while (b <= c && (comparison = comp.compare(array[b], partionValue)) <= 0) {
        if (comparison == 0) {
          temp = array[a];
          array[a++] = array[b];
          array[b] = temp;
        }
        b++;
      }
      while (c >= b && (comparison = comp.compare(array[c], partionValue)) >= 0) {
        if (comparison == 0) {
          temp = array[c];
          array[c] = array[d];
          array[d--] = temp;
        }
        c--;
      }
      if (b > c) {
        break;
      }
      temp = array[b];
      array[b++] = array[c];
      array[c--] = temp;
    }
    length = a - start < b - a ? a - start : b - a;
    int l = start;
    int h = b - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    length = d - c < end - 1 - d ? d - c : end - 1 - d;
    l = b;
    h = end - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    if ((length = b - a) > 0) {
      quickSort0(start, start + length, array, comp);
    }
    if ((length = d - c) > 0) {
      quickSort0(end - length, end, array, comp);
    }
  }

  
  /**
   * Sorts some external data with QuickSort.
   * 
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @param comp
   *          the comparator.
   * @param swap an object that can exchange the positions of two items.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   */
  public static void quickSort(int start, int end, IntComparator comp, Swapper swap) {
    checkBounds(end + 1, start, end);
    quickSort0(start, end, comp, swap);
  }
  
  private static void quickSort0(int start, int end, IntComparator comp, Swapper swap) {
    int length = end - start;
    if (length < 7) {
      insertionSort(start, end, comp, swap);
      return;
    }
    int middle = (start + end) / 2;
    if (length > 7) {
      int bottom = start;
      int top = end - 1;
      if (length > 40) {
        // for lots of data, bottom, middle and top are medians near the beginning, middle or end of the data
        int skosh = length / 8;
        bottom = med3(bottom, bottom + skosh, bottom + (2 * skosh), comp);
        middle = med3(middle - skosh, middle, middle + skosh, comp);
        top = med3(top - (2 * skosh), top - skosh, top, comp);
      }
      middle = med3(bottom, middle, top, comp);
    }

    int partitionIndex = middle; // an index, not a value.
    
    // regions from a to b and from c to d are what we will recursively sort
    int a = start;
    int b = a;
    int c = end - 1;
    int d = c;
    while (b <= c) {
      // copy all values equal to the partition value to before a..b.  In the process, advance b
      // as long as values less than the partition or equal are found, also stop when a..b collides with c..d
      int comparison;
      while (b <= c && (comparison = comp.compare(b, partitionIndex)) <= 0) {
        if (comparison == 0) {
          if (a == partitionIndex) {
            partitionIndex = b;
          } else if (b == partitionIndex) {
            partitionIndex = a;
          }
          swap.swap(a, b);
          a++;
        }
        b++;
      }
      // at this point [start..a) has partition values, [a..b) has values < partition
      // also, either b>c or v[b] > partition value

      while (c >= b && (comparison = comp.compare(c, partitionIndex)) >= 0) {
        if (comparison == 0) {
          if (c == partitionIndex) {
            partitionIndex = d;
          } else if (d == partitionIndex) {
            partitionIndex = c;
          }
          swap.swap(c, d);

          d--;
        }
        c--;
      }
      // now we also know that [d..end] contains partition values,
      // [c..d) contains values > partition value
      // also, either b>c or (v[b] > partition OR v[c] < partition)

      if (b <= c) {
        // v[b] > partition OR v[c] < partition
        // swapping will let us continue to grow the two regions
        if (c == partitionIndex) {
          partitionIndex = b;
        } else if (b == partitionIndex) {
          partitionIndex = d;
        }
        swap.swap(b, c);
        b++;
        c--;
      }
    }
    // now we know
    // b = c+1
    // [start..a) and [d..end) contain partition value
    // all of [a..b) are less than partition
    // all of [c..d) are greater than partition

    // shift [a..b) to beginning
    length = Math.min(a - start, b - a);
    int l = start;
    int h = b - length;
    while (length-- > 0) {
      swap.swap(l, h);
      l++;
      h++;
    }

    // shift [c..d) to end
    length = Math.min(d - c, end - 1 - d);
    l = b;
    h = end - length;
    while (length-- > 0) {
      swap.swap(l, h);
      l++;
      h++;
    }

    // recurse left and right
    length = b - a;
    if (length > 0) {
      quickSort0(start, start + length, comp, swap);
    }

    length = d - c;
    if (length > 0) {
      quickSort0(end - length, end, comp, swap);
    }
  }

  /**
   * In-place insertion sort that is fast for pre-sorted data.
   *
   * @param start Where to start sorting (inclusive)
   * @param end   Where to stop (exclusive)
   * @param comp  Sort order.
   * @param swap  How to swap items.
   */
  private static void insertionSort(int start, int end, IntComparator comp, Swapper swap) {
    for (int i = start + 1; i < end; i++) {
      for (int j = i; j > start && comp.compare(j - 1, j) > 0; j--) {
        swap.swap(j - 1, j);
      }
    }
  }
  /**
   * Sorts the specified range in the array in a specified order.
   * 
   * @param array
   *          the {@code char} array to be sorted.
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   */
  public static void quickSort(char[] array, int start, int end, CharComparator comp) {
    Preconditions.checkNotNull(array);
    checkBounds(array.length, start, end);
    quickSort0(start, end, array, comp);
  }
  
  private static void quickSort0(int start, int end, char[] array, CharComparator comp) {
    char temp;
    int length = end - start;
    if (length < 7) {
      for (int i = start + 1; i < end; i++) {
        for (int j = i; j > start && comp.compare(array[j - 1], array[j]) > 0; j--) {
          temp = array[j];
          array[j] = array[j - 1];
          array[j - 1] = temp;
        }
      }
      return;
    }
    int middle = (start + end) / 2;
    if (length > 7) {
      int bottom = start;
      int top = end - 1;
      if (length > 40) {
        length /= 8;
        bottom = med3(array, bottom, bottom + length, bottom + (2 * length),
            comp);
        middle = med3(array, middle - length, middle, middle + length, comp);
        top = med3(array, top - (2 * length), top - length, top, comp);
      }
      middle = med3(array, bottom, middle, top, comp);
    }
    char partionValue = array[middle];
    int a = start;
    int b = a;
    int c = end - 1;
    int d = c;
    while (true) {
      int comparison;
      while (b <= c && (comparison = comp.compare(array[b], partionValue)) <= 0) {
        if (comparison == 0) {
          temp = array[a];
          array[a++] = array[b];
          array[b] = temp;
        }
        b++;
      }
      while (c >= b && (comparison = comp.compare(array[c], partionValue)) >= 0) {
        if (comparison == 0) {
          temp = array[c];
          array[c] = array[d];
          array[d--] = temp;
        }
        c--;
      }
      if (b > c) {
        break;
      }
      temp = array[b];
      array[b++] = array[c];
      array[c--] = temp;
    }
    length = a - start < b - a ? a - start : b - a;
    int l = start;
    int h = b - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    length = d - c < end - 1 - d ? d - c : end - 1 - d;
    l = b;
    h = end - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    if ((length = b - a) > 0) {
      quickSort0(start, start + length, array, comp);
    }
    if ((length = d - c) > 0) {
      quickSort0(end - length, end, array, comp);
    }
  }
  
  /**
   * Sorts the specified range in the array in a specified order.
   * 
   * @param array
   *          the {@code double} array to be sorted.
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @param comp
   *          the comparison.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   * @see Double#compareTo(Double)
   */
  public static void quickSort(double[] array, int start, int end, DoubleComparator comp) {
    Preconditions.checkNotNull(array);
    checkBounds(array.length, start, end);
    quickSort0(start, end, array, comp);
  }
  
  private static void quickSort0(int start, int end, double[] array, DoubleComparator comp) {
    double temp;
    int length = end - start;
    if (length < 7) {
      for (int i = start + 1; i < end; i++) {
        for (int j = i; j > start && comp.compare(array[j], array[j - 1]) < 0; j--) {
          temp = array[j];
          array[j] = array[j - 1];
          array[j - 1] = temp;
        }
      }
      return;
    }
    int middle = (start + end) / 2;
    if (length > 7) {
      int bottom = start;
      int top = end - 1;
      if (length > 40) {
        length /= 8;
        bottom = med3(array, bottom, bottom + length, bottom + (2 * length),
            comp);
        middle = med3(array, middle - length, middle, middle + length, comp);
        top = med3(array, top - (2 * length), top - length, top, comp);
      }
      middle = med3(array, bottom, middle, top, comp);
    }
    double partionValue = array[middle];
    int a = start;
    int b = a;
    int c = end - 1;
    int d = c;
    while (true) {
      int comparison;
      while (b <= c && (comparison = comp.compare(partionValue, array[b])) >= 0) {
        if (comparison == 0) {
          temp = array[a];
          array[a++] = array[b];
          array[b] = temp;
        }
        b++;
      }
      while (c >= b && (comparison = comp.compare(array[c], partionValue)) >= 0) {
        if (comparison == 0) {
          temp = array[c];
          array[c] = array[d];
          array[d--] = temp;
        }
        c--;
      }
      if (b > c) {
        break;
      }
      temp = array[b];
      array[b++] = array[c];
      array[c--] = temp;
    }
    length = a - start < b - a ? a - start : b - a;
    int l = start;
    int h = b - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    length = d - c < end - 1 - d ? d - c : end - 1 - d;
    l = b;
    h = end - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    if ((length = b - a) > 0) {
      quickSort0(start, start + length, array, comp);
    }
    if ((length = d - c) > 0) {
      quickSort0(end - length, end, array, comp);
    }
  }
  
  /**
   * Sorts the specified range in the array in a specified order.
   * 
   * @param array
   *          the {@code float} array to be sorted.
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @param comp
   *          the comparator.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   */
  public static void quickSort(float[] array, int start, int end, FloatComparator comp) {
    Preconditions.checkNotNull(array);
    checkBounds(array.length, start, end);
    quickSort0(start, end, array, comp);
  }
  
  private static void quickSort0(int start, int end, float[] array, FloatComparator comp) {
    float temp;
    int length = end - start;
    if (length < 7) {
      for (int i = start + 1; i < end; i++) {
        for (int j = i; j > start && comp.compare(array[j], array[j - 1]) < 0; j--) {
          temp = array[j];
          array[j] = array[j - 1];
          array[j - 1] = temp;
        }
      }
      return;
    }
    int middle = (start + end) / 2;
    if (length > 7) {
      int bottom = start;
      int top = end - 1;
      if (length > 40) {
        length /= 8;
        bottom = med3(array, bottom, bottom + length, bottom + (2 * length),
            comp);
        middle = med3(array, middle - length, middle, middle + length, comp);
        top = med3(array, top - (2 * length), top - length, top, comp);
      }
      middle = med3(array, bottom, middle, top, comp);
    }
    float partionValue = array[middle];
    int a = start;
    int b = a;
    int c = end - 1;
    int d = c;
    while (true) {
      int comparison;
      while (b <= c && (comparison = comp.compare(partionValue, array[b])) >= 0) {
        if (comparison == 0) {
          temp = array[a];
          array[a++] = array[b];
          array[b] = temp;
        }
        b++;
      }
      while (c >= b && (comparison = comp.compare(array[c], partionValue)) >= 0) {
        if (comparison == 0) {
          temp = array[c];
          array[c] = array[d];
          array[d--] = temp;
        }
        c--;
      }
      if (b > c) {
        break;
      }
      temp = array[b];
      array[b++] = array[c];
      array[c--] = temp;
    }
    length = a - start < b - a ? a - start : b - a;
    int l = start;
    int h = b - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    length = d - c < end - 1 - d ? d - c : end - 1 - d;
    l = b;
    h = end - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    if ((length = b - a) > 0) {
      quickSort0(start, start + length, array, comp);
    }
    if ((length = d - c) > 0) {
      quickSort0(end - length, end, array, comp);
    }
  }
  
  /**
   * Sorts the specified range in the array in a specified order.
   * 
   * @param array
   *          the {@code int} array to be sorted.
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @param comp
   *          the comparator.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   */
  public static void quickSort(int[] array, int start, int end, IntComparator comp) {
    Preconditions.checkNotNull(array);
    checkBounds(array.length, start, end);
    quickSort0(start, end, array, comp);
  }
  
  private static void quickSort0(int start, int end, int[] array, IntComparator comp) {
    int temp;
    int length = end - start;
    if (length < 7) {
      for (int i = start + 1; i < end; i++) {
        for (int j = i; j > start && comp.compare(array[j - 1], array[j]) > 0; j--) {
          temp = array[j];
          array[j] = array[j - 1];
          array[j - 1] = temp;
        }
      }
      return;
    }
    int middle = (start + end) / 2;
    if (length > 7) {
      int bottom = start;
      int top = end - 1;
      if (length > 40) {
        length /= 8;
        bottom = med3(array, bottom, bottom + length, bottom + (2 * length),
            comp);
        middle = med3(array, middle - length, middle, middle + length, comp);
        top = med3(array, top - (2 * length), top - length, top, comp);
      }
      middle = med3(array, bottom, middle, top, comp);
    }
    int partionValue = array[middle];
    int a = start;
    int b = a;
    int c = end - 1;
    int d = c;
    while (true) {
      int comparison;
      while (b <= c && (comparison = comp.compare(array[b], partionValue)) <= 0) {
        if (comparison == 0) {
          temp = array[a];
          array[a++] = array[b];
          array[b] = temp;
        }
        b++;
      }
      while (c >= b && (comparison = comp.compare(array[c], partionValue)) >= 0) {
        if (comparison == 0) {
          temp = array[c];
          array[c] = array[d];
          array[d--] = temp;
        }
        c--;
      }
      if (b > c) {
        break;
      }
      temp = array[b];
      array[b++] = array[c];
      array[c--] = temp;
    }
    length = a - start < b - a ? a - start : b - a;
    int l = start;
    int h = b - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    length = d - c < end - 1 - d ? d - c : end - 1 - d;
    l = b;
    h = end - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    if ((length = b - a) > 0) {
      quickSort0(start, start + length, array, comp);
    }
    if ((length = d - c) > 0) {
      quickSort0(end - length, end, array, comp);
    }
  }
  
  /**
   * Sorts the specified range in the array in a specified order.
   * 
   * @param array
   *          the {@code long} array to be sorted.
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @param comp
   *          the comparator.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   */
  public static void quickSort(long[] array, int start, int end, LongComparator comp) {
    Preconditions.checkNotNull(array);
    checkBounds(array.length, start, end);
    quickSort0(start, end, array, comp);
  }
  
  private static void quickSort0(int start, int end, long[] array, LongComparator comp) {
    long temp;
    int length = end - start;
    if (length < 7) {
      for (int i = start + 1; i < end; i++) {
        for (int j = i; j > start && comp.compare(array[j - 1], array[j]) > 0; j--) {
          temp = array[j];
          array[j] = array[j - 1];
          array[j - 1] = temp;
        }
      }
      return;
    }
    int middle = (start + end) / 2;
    if (length > 7) {
      int bottom = start;
      int top = end - 1;
      if (length > 40) {
        length /= 8;
        bottom = med3(array, bottom, bottom + length, bottom + (2 * length),
            comp);
        middle = med3(array, middle - length, middle, middle + length, comp);
        top = med3(array, top - (2 * length), top - length, top, comp);
      }
      middle = med3(array, bottom, middle, top, comp);
    }
    long partionValue = array[middle];
    int a = start;
    int b = a;
    int c = end - 1;
    int d = c;
    while (true) {
      int comparison;
      while (b <= c && (comparison = comp.compare(array[b], partionValue)) <= 0) {
        if (comparison == 0) {
          temp = array[a];
          array[a++] = array[b];
          array[b] = temp;
        }
        b++;
      }
      while (c >= b && (comparison = comp.compare(array[c], partionValue)) >= 0) {
        if (comparison == 0) {
          temp = array[c];
          array[c] = array[d];
          array[d--] = temp;
        }
        c--;
      }
      if (b > c) {
        break;
      }
      temp = array[b];
      array[b++] = array[c];
      array[c--] = temp;
    }
    length = a - start < b - a ? a - start : b - a;
    int l = start;
    int h = b - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    length = d - c < end - 1 - d ? d - c : end - 1 - d;
    l = b;
    h = end - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    if ((length = b - a) > 0) {
      quickSort0(start, start + length, array, comp);
    }
    if ((length = d - c) > 0) {
      quickSort0(end - length, end, array, comp);
    }
  }
  
  /**
   * Sorts the specified range in the array in a specified order.
   * 
   * @param array
   *          the array to be sorted.
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @param comp
   *          the comparator.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   */
  public static <T> void quickSort(T[] array, int start, int end, Comparator<T> comp) {
    Preconditions.checkNotNull(array);
    checkBounds(array.length, start, end);
    quickSort0(start, end, array, comp);
  }
  
  private static final class ComparableAdaptor<T extends Comparable<? super T>>
      implements Comparator<T>, Serializable {
    
    @Override
    public int compare(T o1, T o2) {
      return o1.compareTo(o2);
    }
    
  }
  
  /**
   * Sort the specified range of an array of object that implement the Comparable
   * interface.
   * @param <T> The type of object.
   * @param array the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   */
  public static <T extends Comparable<? super T>> void quickSort(T[] array, int start, int end) {
    quickSort(array, start, end, new ComparableAdaptor<T>());
  }
  
  private static <T> void quickSort0(int start, int end, T[] array, Comparator<T> comp) {
    T temp;
    int length = end - start;
    if (length < 7) {
      for (int i = start + 1; i < end; i++) {
        for (int j = i; j > start && comp.compare(array[j - 1], array[j]) > 0; j--) {
          temp = array[j];
          array[j] = array[j - 1];
          array[j - 1] = temp;
        }
      }
      return;
    }
    int middle = (start + end) / 2;
    if (length > 7) {
      int bottom = start;
      int top = end - 1;
      if (length > 40) {
        length /= 8;
        bottom = med3(array, bottom, bottom + length, bottom + (2 * length),
            comp);
        middle = med3(array, middle - length, middle, middle + length, comp);
        top = med3(array, top - (2 * length), top - length, top, comp);
      }
      middle = med3(array, bottom, middle, top, comp);
    }
    T partionValue = array[middle];
    int a = start;
    int b = a;
    int c = end - 1;
    int d = c;
    while (true) {
      int comparison;
      while (b <= c && (comparison = comp.compare(array[b], partionValue)) <= 0) {
        if (comparison == 0) {
          temp = array[a];
          array[a++] = array[b];
          array[b] = temp;
        }
        b++;
      }
      while (c >= b && (comparison = comp.compare(array[c], partionValue)) >= 0) {
        if (comparison == 0) {
          temp = array[c];
          array[c] = array[d];
          array[d--] = temp;
        }
        c--;
      }
      if (b > c) {
        break;
      }
      temp = array[b];
      array[b++] = array[c];
      array[c--] = temp;
    }
    length = a - start < b - a ? a - start : b - a;
    int l = start;
    int h = b - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    length = d - c < end - 1 - d ? d - c : end - 1 - d;
    l = b;
    h = end - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    if ((length = b - a) > 0) {
      quickSort0(start, start + length, array, comp);
    }
    if ((length = d - c) > 0) {
      quickSort0(end - length, end, array, comp);
    }
  }
  
  /**
   * Sorts the specified range in the array in ascending numerical order.
   * 
   * @param array
   *          the {@code short} array to be sorted.
   * @param start
   *          the start index to sort.
   * @param end
   *          the last + 1 index to sort.
   * @throws IllegalArgumentException
   *           if {@code start > end}.
   * @throws ArrayIndexOutOfBoundsException
   *           if {@code start < 0} or {@code end > array.length}.
   */
  public static void quickSort(short[] array, int start, int end, ShortComparator comp) {
    Preconditions.checkNotNull(array);
    checkBounds(array.length, start, end);
    quickSort0(start, end, array, comp);
  }
  
  private static void quickSort0(int start, int end, short[] array, ShortComparator comp) {
    short temp;
    int length = end - start;
    if (length < 7) {
      for (int i = start + 1; i < end; i++) {
        for (int j = i; j > start && comp.compare(array[j - 1], array[j]) > 0; j--) {
          temp = array[j];
          array[j] = array[j - 1];
          array[j - 1] = temp;
        }
      }
      return;
    }
    int middle = (start + end) / 2;
    if (length > 7) {
      int bottom = start;
      int top = end - 1;
      if (length > 40) {
        length /= 8;
        bottom = med3(array, bottom, bottom + length, bottom + (2 * length),
            comp);
        middle = med3(array, middle - length, middle, middle + length, comp);
        top = med3(array, top - (2 * length), top - length, top, comp);
      }
      middle = med3(array, bottom, middle, top, comp);
    }
    short partionValue = array[middle];
    int a = start;
    int b = a;
    int c = end - 1;
    int d = c;
    while (true) {
      int comparison;
      while (b <= c && (comparison = comp.compare(array[b], partionValue)) < 0) {
        if (comparison == 0) {
          temp = array[a];
          array[a++] = array[b];
          array[b] = temp;
        }
        b++;
      }
      while (c >= b && (comparison = comp.compare(array[c], partionValue)) > 0) {
        if (comparison == 0) {
          temp = array[c];
          array[c] = array[d];
          array[d--] = temp;
        }
        c--;
      }
      if (b > c) {
        break;
      }
      temp = array[b];
      array[b++] = array[c];
      array[c--] = temp;
    }
    length = a - start < b - a ? a - start : b - a;
    int l = start;
    int h = b - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    length = d - c < end - 1 - d ? d - c : end - 1 - d;
    l = b;
    h = end - length;
    while (length-- > 0) {
      temp = array[l];
      array[l++] = array[h];
      array[h++] = temp;
    }
    if ((length = b - a) > 0) {
      quickSort0(start, start + length, array, comp);
    }
    if ((length = d - c) > 0) {
      quickSort0(end - length, end, array, comp);
    }
  }

  /**
   * Perform a merge sort on the specified range of an array.
   * 
   * @param <T> the type of object in the array.
   * @param array the array.
   * @param start first index. 
   * @param end last index (exclusive).
   * @param comp comparator object.
   */
  @SuppressWarnings("unchecked") // required to make the temp array work, afaict.
  public static <T> void mergeSort(T[] array, int start, int end, Comparator<T> comp) {
    checkBounds(array.length, start, end);
    int length = end - start;
    if (length <= 0) {
      return;
    }
    
    T[] out = (T[]) new Object[array.length];
    System.arraycopy(array, start, out, start, length);
    mergeSort(out, array, start, end, comp);
  }
  
  /**
   * Perform a merge sort of the specific range of an array of objects that implement
   * Comparable.
   * @param <T> the type of the objects in the array.
   * @param array the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   */
  public static <T extends Comparable<? super T>> void mergeSort(T[] array, int start, int end) {
    mergeSort(array, start, end, new ComparableAdaptor<T>());
  }
  
  /**
   * Performs a sort on the section of the array between the given indices using
   * a mergesort with exponential search algorithm (in which the merge is
   * performed by exponential search). n*log(n) performance is guaranteed and in
   * the average case it will be faster then any mergesort in which the merge is
   * performed by linear search.
   * 
   * @param in
   *          - the array for sorting.
   * @param out
   *          - the result, sorted array.
   * @param start
   *          the start index
   * @param end
   *          the end index + 1
   * @param c
   *          - the comparator to determine the order of the array.
   */
  private static <T> void mergeSort(T[] in, T[] out, int start, int end, Comparator<T> c) {
    int len = end - start;
    // use insertion sort for small arrays
    if (len <= SIMPLE_LENGTH) {
      for (int i = start + 1; i < end; i++) {
        T current = out[i];
        T prev = out[i - 1];
        if (c.compare(prev, current) > 0) {
          int j = i;
          do {
            out[j--] = prev;
          } while (j > start && (c.compare(prev = out[j - 1], current) > 0));
          out[j] = current;
        }
      }
      return;
    }
    int med = (end + start) >>> 1;
    mergeSort(out, in, start, med, c);
    mergeSort(out, in, med, end, c);
    
    // merging
    
    // if arrays are already sorted - no merge
    if (c.compare(in[med - 1], in[med]) <= 0) {
      System.arraycopy(in, start, out, start, len);
      return;
    }
    int r = med;
    int i = start;

    // use merging with exponential search
    do {
      T fromVal = in[start];
      T rVal = in[r];
      if (c.compare(fromVal, rVal) <= 0) {
        int l_1 = find(in, rVal, -1, start + 1, med - 1, c);
        int toCopy = l_1 - start + 1;
        System.arraycopy(in, start, out, i, toCopy);
        i += toCopy;
        out[i++] = rVal;
        r++;
        start = l_1 + 1;
      } else {
        int r_1 = find(in, fromVal, 0, r + 1, end - 1, c);
        int toCopy = r_1 - r + 1;
        System.arraycopy(in, r, out, i, toCopy);
        i += toCopy;
        out[i++] = fromVal;
        start++;
        r = r_1 + 1;
      }
    } while ((end - r) > 0 && (med - start) > 0);
    
    // copy rest of array
    if ((end - r) <= 0) {
      System.arraycopy(in, start, out, i, med - start);
    } else {
      System.arraycopy(in, r, out, i, end - r);
    }
  }
  
  /**
   * Finds the place of specified range of specified sorted array, where the
   * element should be inserted for getting sorted array. Uses exponential
   * search algorithm.
   * 
   * @param arr
   *          - the array with already sorted range
   * @param val
   *          - object to be inserted
   * @param l
   *          - the start index
   * @param r
   *          - the end index
   * @param bnd
   *          - possible values 0,-1. "-1" - val is located at index more then
   *          elements equals to val. "0" - val is located at index less then
   *          elements equals to val.
   * @param c
   *          - the comparator used to compare Objects
   */
  private static <T> int find(T[] arr, T val, int bnd, int l, int r, Comparator<T> c) {
    int m = l;
    int d = 1;
    while (m <= r) {
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
        break;
      }
      m += d;
      d <<= 1;
    }
    while (l <= r) {
      m = (l + r) >>> 1;
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
      }
    }
    return l - 1;
  }
  
  private static final ByteComparator NATURAL_BYTE_COMPARISON = new ByteComparator() {
    @Override
    public int compare(byte o1, byte o2) {
      return o1 - o2;
    }
  };
    
    /**
     * Perform a merge sort on a range of a byte array, using numerical order.
     * @param array the array.
     * @param start the first index.
     * @param end the last index (exclusive).
     */
  public static void mergeSort(byte[] array, int start, int end) {
    mergeSort(array, start, end, NATURAL_BYTE_COMPARISON);
  }
  
  /**
   * Perform a merge sort on a range of a byte array using a specified ordering.
   * @param array the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   * @param comp the comparator object.
   */
  public static void mergeSort(byte[] array, int start, int end, ByteComparator comp) {
    checkBounds(array.length, start, end);
    byte[] out = Arrays.copyOf(array, array.length);
    mergeSort(out, array, start, end, comp);
  }

  private static void mergeSort(byte[] in, byte[] out, int start, int end, ByteComparator c) {
    int len = end - start;
    // use insertion sort for small arrays
    if (len <= SIMPLE_LENGTH) {
      for (int i = start + 1; i < end; i++) {
        byte current = out[i];
        byte prev = out[i - 1];
        if (c.compare(prev, current) > 0) {
          int j = i;
          do {
            out[j--] = prev;
          } while (j > start && (c.compare(prev = out[j - 1], current) > 0));
          out[j] = current;
        }
      }
      return;
    }
    int med = (end + start) >>> 1;
    mergeSort(out, in, start, med, c);
    mergeSort(out, in, med, end, c);
    
    // merging
    
    // if arrays are already sorted - no merge
    if (c.compare(in[med - 1], in[med]) <= 0) {
      System.arraycopy(in, start, out, start, len);
      return;
    }
    int r = med;
    int i = start;

    // use merging with exponential search
    do {
      byte fromVal = in[start];
      byte rVal = in[r];
      if (c.compare(fromVal, rVal) <= 0) {
        int l_1 = find(in, rVal, -1, start + 1, med - 1, c);
        int toCopy = l_1 - start + 1;
        System.arraycopy(in, start, out, i, toCopy);
        i += toCopy;
        out[i++] = rVal;
        r++;
        start = l_1 + 1;
      } else {
        int r_1 = find(in, fromVal, 0, r + 1, end - 1, c);
        int toCopy = r_1 - r + 1;
        System.arraycopy(in, r, out, i, toCopy);
        i += toCopy;
        out[i++] = fromVal;
        start++;
        r = r_1 + 1;
      }
    } while ((end - r) > 0 && (med - start) > 0);
    
    // copy rest of array
    if ((end - r) <= 0) {
      System.arraycopy(in, start, out, i, med - start);
    } else {
      System.arraycopy(in, r, out, i, end - r);
    }
  }

  private static int find(byte[] arr, byte val, int bnd, int l, int r, ByteComparator c) {
    int m = l;
    int d = 1;
    while (m <= r) {
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
        break;
      }
      m += d;
      d <<= 1;
    }
    while (l <= r) {
      m = (l + r) >>> 1;
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
      }
    }
    return l - 1;
  }
  
  private static final CharComparator NATURAL_CHAR_COMPARISON = new CharComparator() {
    @Override
    public int compare(char o1, char o2) {
      return o1 - o2;
    }
  };
    
    /**
     * Perform a merge sort on a range of a char array, using numerical order.
     * @param array the array.
     * @param start the first index.
     * @param end the last index (exclusive).
     */
  public static void mergeSort(char[] array, int start, int end) {
    mergeSort(array, start, end, NATURAL_CHAR_COMPARISON);
  }

  /**
   * Perform a merge sort on a range of a char array using a specified ordering.
   * @param array the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   * @param comp the comparator object.
   */
  public static void mergeSort(char[] array, int start, int end, CharComparator comp) {
    checkBounds(array.length, start, end);
    char[] out = Arrays.copyOf(array, array.length);
    mergeSort(out, array, start, end, comp);
  }

  private static void mergeSort(char[] in, char[] out, int start, int end, CharComparator c) {
    int len = end - start;
    // use insertion sort for small arrays
    if (len <= SIMPLE_LENGTH) {
      for (int i = start + 1; i < end; i++) {
        char current = out[i];
        char prev = out[i - 1];
        if (c.compare(prev, current) > 0) {
          int j = i;
          do {
            out[j--] = prev;
          } while (j > start && (c.compare(prev = out[j - 1], current) > 0));
          out[j] = current;
        }
      }
      return;
    }
    int med = (end + start) >>> 1;
    mergeSort(out, in, start, med, c);
    mergeSort(out, in, med, end, c);
    
    // merging
    
    // if arrays are already sorted - no merge
    if (c.compare(in[med - 1], in[med]) <= 0) {
      System.arraycopy(in, start, out, start, len);
      return;
    }
    int r = med;
    int i = start;

    // use merging with exponential search
    do {
      char fromVal = in[start];
      char rVal = in[r];
      if (c.compare(fromVal, rVal) <= 0) {
        int l_1 = find(in, rVal, -1, start + 1, med - 1, c);
        int toCopy = l_1 - start + 1;
        System.arraycopy(in, start, out, i, toCopy);
        i += toCopy;
        out[i++] = rVal;
        r++;
        start = l_1 + 1;
      } else {
        int r_1 = find(in, fromVal, 0, r + 1, end - 1, c);
        int toCopy = r_1 - r + 1;
        System.arraycopy(in, r, out, i, toCopy);
        i += toCopy;
        out[i++] = fromVal;
        start++;
        r = r_1 + 1;
      }
    } while ((end - r) > 0 && (med - start) > 0);
    
    // copy rest of array
    if ((end - r) <= 0) {
      System.arraycopy(in, start, out, i, med - start);
    } else {
      System.arraycopy(in, r, out, i, end - r);
    }
  }

  private static int find(char[] arr, char val, int bnd, int l, int r, CharComparator c) {
    int m = l;
    int d = 1;
    while (m <= r) {
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
        break;
      }
      m += d;
      d <<= 1;
    }
    while (l <= r) {
      m = (l + r) >>> 1;
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
      }
    }
    return l - 1;
  }
  
  private static final ShortComparator NATURAL_SHORT_COMPARISON = new ShortComparator() {
    @Override
    public int compare(short o1, short o2) {
      return o1 - o2;
    }
  };
    
    /**
     * Perform a merge sort on a range of a short array, using numerical order.
     * @param array the array.
     * @param start the first index.
     * @param end the last index (exclusive).
     */
  public static void mergeSort(short[] array, int start, int end) {
    mergeSort(array, start, end, NATURAL_SHORT_COMPARISON);
  }
  
  public static void mergeSort(short[] array, int start, int end, ShortComparator comp) {
    checkBounds(array.length, start, end);
    short[] out = Arrays.copyOf(array, array.length);
    mergeSort(out, array, start, end, comp);
  }

  
  /**
   * Perform a merge sort on a range of a short array using a specified ordering.
   * @param in the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   * @param c the comparator object.
   */
  private static void mergeSort(short[] in, short[] out, int start, int end, ShortComparator c) {
    int len = end - start;
    // use insertion sort for small arrays
    if (len <= SIMPLE_LENGTH) {
      for (int i = start + 1; i < end; i++) {
        short current = out[i];
        short prev = out[i - 1];
        if (c.compare(prev, current) > 0) {
          int j = i;
          do {
            out[j--] = prev;
          } while (j > start && (c.compare(prev = out[j - 1], current) > 0));
          out[j] = current;
        }
      }
      return;
    }
    int med = (end + start) >>> 1;
    mergeSort(out, in, start, med, c);
    mergeSort(out, in, med, end, c);
    
    // merging
    
    // if arrays are already sorted - no merge
    if (c.compare(in[med - 1], in[med]) <= 0) {
      System.arraycopy(in, start, out, start, len);
      return;
    }
    int r = med;
    int i = start;

    // use merging with exponential search
    do {
      short fromVal = in[start];
      short rVal = in[r];
      if (c.compare(fromVal, rVal) <= 0) {
        int l_1 = find(in, rVal, -1, start + 1, med - 1, c);
        int toCopy = l_1 - start + 1;
        System.arraycopy(in, start, out, i, toCopy);
        i += toCopy;
        out[i++] = rVal;
        r++;
        start = l_1 + 1;
      } else {
        int r_1 = find(in, fromVal, 0, r + 1, end - 1, c);
        int toCopy = r_1 - r + 1;
        System.arraycopy(in, r, out, i, toCopy);
        i += toCopy;
        out[i++] = fromVal;
        start++;
        r = r_1 + 1;
      }
    } while ((end - r) > 0 && (med - start) > 0);
    
    // copy rest of array
    if ((end - r) <= 0) {
      System.arraycopy(in, start, out, i, med - start);
    } else {
      System.arraycopy(in, r, out, i, end - r);
    }
  }

  private static int find(short[] arr, short val, int bnd, int l, int r, ShortComparator c) {
    int m = l;
    int d = 1;
    while (m <= r) {
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
        break;
      }
      m += d;
      d <<= 1;
    }
    while (l <= r) {
      m = (l + r) >>> 1;
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
      }
    }
    return l - 1;
  }
  
  private static final IntComparator NATURAL_INT_COMPARISON = new IntComparator() {
    @Override
    public int compare(int o1, int o2) {
      return o1 < o2 ? -1 : o1 > o2 ? 1 : 0;
    }
  };
    
  public static void mergeSort(int[] array, int start, int end) {
    mergeSort(array, start, end, NATURAL_INT_COMPARISON);
  }

  /**
   * Perform a merge sort on a range of a int array using numerical order.
   * @param array the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   * @param comp the comparator object.
   */
  public static void mergeSort(int[] array, int start, int end, IntComparator comp) {
    checkBounds(array.length, start, end);
    int[] out = Arrays.copyOf(array, array.length);
    mergeSort(out, array, start, end, comp);
  }

  /**
   * Perform a merge sort on a range of a int array using a specified ordering.
   * @param in the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   * @param c the comparator object.
   */
  private static void mergeSort(int[] in, int[] out, int start, int end, IntComparator c) {
    int len = end - start;
    // use insertion sort for small arrays
    if (len <= SIMPLE_LENGTH) {
      for (int i = start + 1; i < end; i++) {
        int current = out[i];
        int prev = out[i - 1];
        if (c.compare(prev, current) > 0) {
          int j = i;
          do {
            out[j--] = prev;
          } while (j > start && (c.compare(prev = out[j - 1], current) > 0));
          out[j] = current;
        }
      }
      return;
    }
    int med = (end + start) >>> 1;
    mergeSort(out, in, start, med, c);
    mergeSort(out, in, med, end, c);
    
    // merging
    
    // if arrays are already sorted - no merge
    if (c.compare(in[med - 1], in[med]) <= 0) {
      System.arraycopy(in, start, out, start, len);
      return;
    }
    int r = med;
    int i = start;

    // use merging with exponential search
    do {
      int fromVal = in[start];
      int rVal = in[r];
      if (c.compare(fromVal, rVal) <= 0) {
        int l_1 = find(in, rVal, -1, start + 1, med - 1, c);
        int toCopy = l_1 - start + 1;
        System.arraycopy(in, start, out, i, toCopy);
        i += toCopy;
        out[i++] = rVal;
        r++;
        start = l_1 + 1;
      } else {
        int r_1 = find(in, fromVal, 0, r + 1, end - 1, c);
        int toCopy = r_1 - r + 1;
        System.arraycopy(in, r, out, i, toCopy);
        i += toCopy;
        out[i++] = fromVal;
        start++;
        r = r_1 + 1;
      }
    } while ((end - r) > 0 && (med - start) > 0);
    
    // copy rest of array
    if ((end - r) <= 0) {
      System.arraycopy(in, start, out, i, med - start);
    } else {
      System.arraycopy(in, r, out, i, end - r);
    }
  }

  private static int find(int[] arr, int val, int bnd, int l, int r, IntComparator c) {
    int m = l;
    int d = 1;
    while (m <= r) {
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
        break;
      }
      m += d;
      d <<= 1;
    }
    while (l <= r) {
      m = (l + r) >>> 1;
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
      }
    }
    return l - 1;
  }
  
  
  private static final LongComparator NATURAL_LONG_COMPARISON = new LongComparator() {
    @Override
    public int compare(long o1, long o2) {
      return o1 < o2 ? -1 : o1 > o2 ? 1 : 0;
    }
  };
    
    /**
     * Perform a merge sort on a range of a long array using numerical order.
     * @param array the array.
     * @param start the first index.
     * @param end the last index (exclusive).
     */
  public static void mergeSort(long[] array, int start, int end) {
    mergeSort(array, start, end, NATURAL_LONG_COMPARISON);
  }

  /**
   * Perform a merge sort on a range of a long array using a specified ordering.
   * @param array the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   * @param comp the comparator object.
   */
  public static void mergeSort(long[] array, int start, int end, LongComparator comp) {
    checkBounds(array.length, start, end);
    long[] out = Arrays.copyOf(array, array.length);
    mergeSort(out, array, start, end, comp);
  }

  private static void mergeSort(long[] in, long[] out, int start, int end, LongComparator c) {
    int len = end - start;
    // use insertion sort for small arrays
    if (len <= SIMPLE_LENGTH) {
      for (int i = start + 1; i < end; i++) {
        long current = out[i];
        long prev = out[i - 1];
        if (c.compare(prev, current) > 0) {
          int j = i;
          do {
            out[j--] = prev;
          } while (j > start && (c.compare(prev = out[j - 1], current) > 0));
          out[j] = current;
        }
      }
      return;
    }
    int med = (end + start) >>> 1;
    mergeSort(out, in, start, med, c);
    mergeSort(out, in, med, end, c);
    
    // merging
    
    // if arrays are already sorted - no merge
    if (c.compare(in[med - 1], in[med]) <= 0) {
      System.arraycopy(in, start, out, start, len);
      return;
    }
    int r = med;
    int i = start;

    // use merging with exponential search
    do {
      long fromVal = in[start];
      long rVal = in[r];
      if (c.compare(fromVal, rVal) <= 0) {
        int l_1 = find(in, rVal, -1, start + 1, med - 1, c);
        int toCopy = l_1 - start + 1;
        System.arraycopy(in, start, out, i, toCopy);
        i += toCopy;
        out[i++] = rVal;
        r++;
        start = l_1 + 1;
      } else {
        int r_1 = find(in, fromVal, 0, r + 1, end - 1, c);
        int toCopy = r_1 - r + 1;
        System.arraycopy(in, r, out, i, toCopy);
        i += toCopy;
        out[i++] = fromVal;
        start++;
        r = r_1 + 1;
      }
    } while ((end - r) > 0 && (med - start) > 0);
    
    // copy rest of array
    if ((end - r) <= 0) {
      System.arraycopy(in, start, out, i, med - start);
    } else {
      System.arraycopy(in, r, out, i, end - r);
    }
  }

  private static int find(long[] arr, long val, int bnd, int l, int r, LongComparator c) {
    int m = l;
    int d = 1;
    while (m <= r) {
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
        break;
      }
      m += d;
      d <<= 1;
    }
    while (l <= r) {
      m = (l + r) >>> 1;
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
      }
    }
    return l - 1;
  }
  
  private static final FloatComparator NATURAL_FLOAT_COMPARISON = new FloatComparator() {
    @Override
    public int compare(float o1, float o2) {
      return Float.compare(o1, o2);
    }
  };
    
    /**
     * Perform a merge sort on a range of a float array using Float.compare for an ordering.
     * @param array the array.
     * @param start the first index.
     * @param end the last index (exclusive).
     */
  public static void mergeSort(float[] array, int start, int end) {
    mergeSort(array, start, end, NATURAL_FLOAT_COMPARISON);
  }

  /**
   * Perform a merge sort on a range of a float array using a specified ordering.
   * @param array the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   * @param comp the comparator object.
   */
  public static void mergeSort(float[] array, int start, int end, FloatComparator comp) {
    checkBounds(array.length, start, end);
    float[] out = Arrays.copyOf(array, array.length);
    mergeSort(out, array, start, end, comp);
  }

  private static void mergeSort(float[] in, float[] out, int start, int end, FloatComparator c) {
    int len = end - start;
    // use insertion sort for small arrays
    if (len <= SIMPLE_LENGTH) {
      for (int i = start + 1; i < end; i++) {
        float current = out[i];
        float prev = out[i - 1];
        if (c.compare(prev, current) > 0) {
          int j = i;
          do {
            out[j--] = prev;
          } while (j > start && (c.compare(prev = out[j - 1], current) > 0));
          out[j] = current;
        }
      }
      return;
    }
    int med = (end + start) >>> 1;
    mergeSort(out, in, start, med, c);
    mergeSort(out, in, med, end, c);
    
    // merging
    
    // if arrays are already sorted - no merge
    if (c.compare(in[med - 1], in[med]) <= 0) {
      System.arraycopy(in, start, out, start, len);
      return;
    }
    int r = med;
    int i = start;

    // use merging with exponential search
    do {
      float fromVal = in[start];
      float rVal = in[r];
      if (c.compare(fromVal, rVal) <= 0) {
        int l_1 = find(in, rVal, -1, start + 1, med - 1, c);
        int toCopy = l_1 - start + 1;
        System.arraycopy(in, start, out, i, toCopy);
        i += toCopy;
        out[i++] = rVal;
        r++;
        start = l_1 + 1;
      } else {
        int r_1 = find(in, fromVal, 0, r + 1, end - 1, c);
        int toCopy = r_1 - r + 1;
        System.arraycopy(in, r, out, i, toCopy);
        i += toCopy;
        out[i++] = fromVal;
        start++;
        r = r_1 + 1;
      }
    } while ((end - r) > 0 && (med - start) > 0);
    
    // copy rest of array
    if ((end - r) <= 0) {
      System.arraycopy(in, start, out, i, med - start);
    } else {
      System.arraycopy(in, r, out, i, end - r);
    }
  }

  private static int find(float[] arr, float val, int bnd, int l, int r, FloatComparator c) {
    int m = l;
    int d = 1;
    while (m <= r) {
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
        break;
      }
      m += d;
      d <<= 1;
    }
    while (l <= r) {
      m = (l + r) >>> 1;
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
      }
    }
    return l - 1;
  }
  
  private static final DoubleComparator NATURAL_DOUBLE_COMPARISON = new DoubleComparator() {
    @Override
    public int compare(double o1, double o2) {
      return Double.compare(o1, o2);
    }
  };
    
    
    /**
     * Perform a merge sort on a range of a double array using a Double.compare as an ordering.
     * @param array the array.
     * @param start the first index.
     * @param end the last index (exclusive).
     */
  public static void mergeSort(double[] array, int start, int end) {
    mergeSort(array, start, end, NATURAL_DOUBLE_COMPARISON);
  }

  /**
   * Perform a merge sort on a range of a double array using a specified ordering.
   * @param array the array.
   * @param start the first index.
   * @param end the last index (exclusive).
   * @param comp the comparator object.
   */
  public static void mergeSort(double[] array, int start, int end, DoubleComparator comp) {
    checkBounds(array.length, start, end);
    double[] out = Arrays.copyOf(array, array.length);
    mergeSort(out, array, start, end, comp);
  }

  private static void mergeSort(double[] in, double[] out, int start, int end, DoubleComparator c) {
    int len = end - start;
    // use insertion sort for small arrays
    if (len <= SIMPLE_LENGTH) {
      for (int i = start + 1; i < end; i++) {
        double current = out[i];
        double prev = out[i - 1];
        if (c.compare(prev, current) > 0) {
          int j = i;
          do {
            out[j--] = prev;
          } while (j > start && (c.compare(prev = out[j - 1], current) > 0));
          out[j] = current;
        }
      }
      return;
    }
    int med = (end + start) >>> 1;
    mergeSort(out, in, start, med, c);
    mergeSort(out, in, med, end, c);
    
    // merging
    
    // if arrays are already sorted - no merge
    if (c.compare(in[med - 1], in[med]) <= 0) {
      System.arraycopy(in, start, out, start, len);
      return;
    }
    int r = med;
    int i = start;

    // use merging with exponential search
    do {
      double fromVal = in[start];
      double rVal = in[r];
      if (c.compare(fromVal, rVal) <= 0) {
        int l_1 = find(in, rVal, -1, start + 1, med - 1, c);
        int toCopy = l_1 - start + 1;
        System.arraycopy(in, start, out, i, toCopy);
        i += toCopy;
        out[i++] = rVal;
        r++;
        start = l_1 + 1;
      } else {
        int r_1 = find(in, fromVal, 0, r + 1, end - 1, c);
        int toCopy = r_1 - r + 1;
        System.arraycopy(in, r, out, i, toCopy);
        i += toCopy;
        out[i++] = fromVal;
        start++;
        r = r_1 + 1;
      }
    } while ((end - r) > 0 && (med - start) > 0);
    
    // copy rest of array
    if ((end - r) <= 0) {
      System.arraycopy(in, start, out, i, med - start);
    } else {
      System.arraycopy(in, r, out, i, end - r);
    }
  }

  private static int find(double[] arr, double val, int bnd, int l, int r, DoubleComparator c) {
    int m = l;
    int d = 1;
    while (m <= r) {
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
        break;
      }
      m += d;
      d <<= 1;
    }
    while (l <= r) {
      m = (l + r) >>> 1;
      if (c.compare(val, arr[m]) > bnd) {
        l = m + 1;
      } else {
        r = m - 1;
      }
    }
    return l - 1;
  }

  /**
   * Transforms two consecutive sorted ranges into a single sorted range. The initial ranges are {@code [first,}
   * middle)</code> and {@code [middle, last)}, and the resulting range is {@code [first, last)}. Elements in
   * the first input range will precede equal elements in the second.
   */
  static void inplaceMerge(int first, int middle, int last, IntComparator comp, Swapper swapper) {
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
      secondCut = lowerBound(middle, last, firstCut, comp);
    } else {
      secondCut = middle + (last - middle) / 2;
      firstCut = upperBound(first, middle, secondCut, comp);
    }
  
    // rotate(firstCut, middle, secondCut, swapper);
    // is manually inlined for speed (jitter inlining seems to work only for small call depths, even if methods
    // are "static private")
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
    inplaceMerge(first, firstCut, middle, comp, swapper);
    inplaceMerge(middle, secondCut, last, comp, swapper);
  }

  /**
   * Performs a binary search on an already-sorted range: finds the first position where an element can be inserted
   * without violating the ordering. Sorting is by a user-supplied comparison function.
   *
   * @param first Beginning of the range.
   * @param last  One past the end of the range.
   * @param x     Element to be searched for.
   * @param comp  Comparison function.
   * @return The largest index i such that, for every j in the range <code>[first, i)</code>,
   *        <code></code></codeA>{@code comp.apply(array[j], x)</code> is {@code true}.
   * @see Sorting#upperBound
   */
  static int lowerBound(int first, int last, int x, IntComparator comp) {
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
      Hence, in contrast to the JDK mergesorts this is an "in-place" mergesort, i.e. does not allocate any temporary
      arrays.
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
    inplaceMerge(fromIndex, mid, toIndex, c, swapper);
  }

  /**
   * Performs a binary search on an already-sorted range: finds the last position where an element can be inserted
   * without violating the ordering. Sorting is by a user-supplied comparison function.
   *
   * @param first Beginning of the range.
   * @param last  One past the end of the range.
   * @param x     Element to be searched for.
   * @param comp  Comparison function.
   * @return The largest index i such that, for every j in the range <code>[first, i)</code>, {@code comp.apply(x,}
   *         array[j])</code> is {@code false}.
   * @see Sorting#lowerBound
   */
  static int upperBound(int first, int last, int x, IntComparator comp) {
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
