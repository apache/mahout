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

import java.util.Comparator;

public final class BinarySearch {

  private BinarySearch() {}

  /**
   * Performs a binary search for the specified element in the specified
   * ascending sorted array. Searching in an unsorted array has an undefined
   * result. It's also undefined which element is found if there are multiple
   * occurrences of the same element.
   *
   * @param array
   *          the sorted {@code byte} array to search.
   * @param value
   *          the {@code byte} element to find.
   * @param from
   *          the first index to sort, inclusive.
   * @param to
   *          the last index to sort, inclusive.
   * @return the non-negative index of the element, or a negative index which is
   *         {@code -index - 1} where the element would be inserted.
   */
  public static int binarySearchFromTo(byte[] array, byte value, int from, int to) {
    int mid = -1;
    while (from <= to) {
      mid = (from + to) >>> 1;
      if (value > array[mid]) {
        from = mid + 1;
      } else if (value == array[mid]) {
        return mid;
      } else {
        to = mid - 1;
      }
    }
    if (mid < 0) {
      return -1;
    }

    return -mid - (value < array[mid] ? 1 : 2);
  }

  /**
   * Performs a binary search for the specified element in the specified
   * ascending sorted array. Searching in an unsorted array has an undefined
   * result. It's also undefined which element is found if there are multiple
   * occurrences of the same element.
   *
   * @param array
   *          the sorted {@code char} array to search.
   * @param value
   *          the {@code char} element to find.
   * @param from
   *          the first index to sort, inclusive.
   * @param to
   *          the last index to sort, inclusive.
   * @return the non-negative index of the element, or a negative index which is
   *         {@code -index - 1} where the element would be inserted.
   */
  public static int binarySearchFromTo(char[] array, char value, int from, int to) {
    int mid = -1;
    while (from <= to) {
      mid = (from + to) >>> 1;
      if (value > array[mid]) {
        from = mid + 1;
      } else if (value == array[mid]) {
        return mid;
      } else {
        to = mid - 1;
      }
    }
    if (mid < 0) {
      return -1;
    }
    return -mid - (value < array[mid] ? 1 : 2);
  }

  /**
   * Performs a binary search for the specified element in the specified
   * ascending sorted array. Searching in an unsorted array has an undefined
   * result. It's also undefined which element is found if there are multiple
   * occurrences of the same element.
   *
   * @param array
   *          the sorted {@code double} array to search.
   * @param value
   *          the {@code double} element to find.
   * @param from
   *          the first index to sort, inclusive.
   * @param to
   *          the last index to sort, inclusive.
   * @return the non-negative index of the element, or a negative index which is
   *         {@code -index - 1} where the element would be inserted.
   */
  public static int binarySearchFromTo(double[] array, double value, int from, int to) {
    long longBits = Double.doubleToLongBits(value);
    int mid = -1;
    while (from <= to) {
      mid = (from + to) >>> 1;
      if (lessThan(array[mid], value)) {
        from = mid + 1;
      } else if (longBits == Double.doubleToLongBits(array[mid])) {
        return mid;
      } else {
        to = mid - 1;
      }
    }
    if (mid < 0) {
      return -1;
    }
    return -mid - (lessThan(value, array[mid]) ? 1 : 2);
  }

  /**
   * Performs a binary search for the specified element in the specified
   * ascending sorted array. Searching in an unsorted array has an undefined
   * result. It's also undefined which element is found if there are multiple
   * occurrences of the same element.
   *
   * @param array
   *          the sorted {@code float} array to search.
   * @param value
   *          the {@code float} element to find.
   * @param from
   *          the first index to sort, inclusive.
   * @param to
   *          the last index to sort, inclusive.
   * @return the non-negative index of the element, or a negative index which is
   *         {@code -index - 1} where the element would be inserted.
   */
  public static int binarySearchFromTo(float[] array, float value, int from, int to) {
    int intBits = Float.floatToIntBits(value);
    int mid = -1;
    while (from <= to) {
      mid = (from + to) >>> 1;
      if (lessThan(array[mid], value)) {
        from = mid + 1;
      } else if (intBits == Float.floatToIntBits(array[mid])) {
        return mid;
      } else {
        to = mid - 1;
      }
    }
    if (mid < 0) {
      return -1;
    }
    return -mid - (lessThan(value, array[mid]) ? 1 : 2);
  }

  /**
   * Performs a binary search for the specified element in the specified
   * ascending sorted array. Searching in an unsorted array has an undefined
   * result. It's also undefined which element is found if there are multiple
   * occurrences of the same element.
   *
   * @param array
   *          the sorted {@code int} array to search.
   * @param value
   *          the {@code int} element to find.
   * @param from
   *          the first index to sort, inclusive.
   * @param to
   *          the last index to sort, inclusive.
   * @return the non-negative index of the element, or a negative index which is
   *         {@code -index - 1} where the element would be inserted.
   */
  public static int binarySearchFromTo(int[] array, int value, int from, int to) {
    int mid = -1;
    while (from <= to) {
      mid = (from + to) >>> 1;
      if (value > array[mid]) {
        from = mid + 1;
      } else if (value == array[mid]) {
        return mid;
      } else {
        to = mid - 1;
      }
    }
    if (mid < 0) {
      return -1;
    }
    return -mid - (value < array[mid] ? 1 : 2);
  }

  /**
   * Performs a binary search for the specified element in the specified
   * ascending sorted array. Searching in an unsorted array has an undefined
   * result. It's also undefined which element is found if there are multiple
   * occurrences of the same element.
   *
   * @param array
   *          the sorted {@code long} array to search.
   * @param value
   *          the {@code long} element to find.
   * @param from
   *          the first index to sort, inclusive.
   * @param to
   *          the last index to sort, inclusive.
   * @return the non-negative index of the element, or a negative index which is
   *         {@code -index - 1} where the element would be inserted.
   */
  public static int binarySearchFromTo(long[] array, long value, int from, int to) {
    int mid = -1;
    while (from <= to) {
      mid = (from + to) >>> 1;
      if (value > array[mid]) {
        from = mid + 1;
      } else if (value == array[mid]) {
        return mid;
      } else {
        to = mid - 1;
      }
    }
    if (mid < 0) {
      return -1;
    }
    return -mid - (value < array[mid] ? 1 : 2);
  }

  /**
   * Performs a binary search for the specified element in the specified
   * ascending sorted array. Searching in an unsorted array has an undefined
   * result. It's also undefined which element is found if there are multiple
   * occurrences of the same element.
   *
   * @param array
   *          the sorted {@code Object} array to search.
   * @param object
   *          the {@code Object} element to find
   * @param from
   *          the first index to sort, inclusive.
   * @param to
   *          the last index to sort, inclusive.
   * @return the non-negative index of the element, or a negative index which is
   *         {@code -index - 1} where the element would be inserted.
   *
   */
  public static <T extends Comparable<T>> int binarySearchFromTo(T[] array, T object, int from, int to) {
    if (array.length == 0) {
      return -1;
    }

    int mid = 0;
    int result = 0;
    while (from <= to) {
      mid = (from + to) >>> 1;
      if ((result = array[mid].compareTo(object)) < 0) {
        from = mid + 1;
      } else if (result == 0) {
        return mid;
      } else {
        to = mid - 1;
      }
    }
    return -mid - (result >= 0 ? 1 : 2);
  }

  /**
   * Performs a binary search for the specified element in the specified
   * ascending sorted array using the {@code Comparator} to compare elements.
   * Searching in an unsorted array has an undefined result. It's also undefined
   * which element is found if there are multiple occurrences of the same
   * element.
   *
   * @param array
   *          the sorted array to search
   * @param object
   *          the element to find
   * @param from
   *          the first index to sort, inclusive.
   * @param to
   *          the last index to sort, inclusive.
   * @param comparator
   *          the {@code Comparator} used to compare the elements.
   * @return the non-negative index of the element, or a negative index which
   */
  public static <T> int binarySearchFromTo(T[] array, T object, int from, int to, Comparator<? super T> comparator) {
    int mid = 0;
    int result = 0;
    while (from <= to) {
      mid = (from + to) >>> 1;
      if ((result = comparator.compare(array[mid], object)) < 0) {
        from = mid + 1;
      } else if (result == 0) {
        return mid;
      } else {
        to = mid - 1;
      }
    }
    return -mid - (result >= 0 ? 1 : 2);
  }

  /**
   * Performs a binary search for the specified element in the specified
   * ascending sorted array. Searching in an unsorted array has an undefined
   * result. It's also undefined which element is found if there are multiple
   * occurrences of the same element.
   *
   * @param array
   *          the sorted {@code short} array to search.
   * @param value
   *          the {@code short} element to find.
   * @param from
   *          the first index to sort, inclusive.
   * @param to
   *          the last index to sort, inclusive.
   * @return the non-negative index of the element, or a negative index which is
   *         {@code -index - 1} where the element would be inserted.
   */
  public static int binarySearchFromTo(short[] array, short value, int from, int to) {
    int mid = -1;
    while (from <= to) {
      mid = (from + to) >>> 1;
      if (value > array[mid]) {
        from = mid + 1;
      } else if (value == array[mid]) {
        return mid;
      } else {
        to = mid - 1;
      }
    }
    if (mid < 0) {
      return -1;
    }
    return -mid - (value < array[mid] ? 1 : 2);
  }

  private static boolean lessThan(double double1, double double2) {
    // A slightly specialized version of
    // Double.compare(double1, double2) < 0.

    // Non-zero and non-NaN checking.
    if (double1 < double2) {
      return true;
    }
    if (double1 > double2) {
      return false;
    }
    if (double1 == double2 && double1 != 0.0) {
      return false;
    }

    // NaNs are equal to other NaNs and larger than any other double.
    if (Double.isNaN(double1)) {
      return false;
    }
    if (Double.isNaN(double2)) {
      return true;
    }

    // Deal with +0.0 and -0.0.
    long d1 = Double.doubleToRawLongBits(double1);
    long d2 = Double.doubleToRawLongBits(double2);
    return d1 < d2;
  }

  private static boolean lessThan(float float1, float float2) {
    // A slightly specialized version of Float.compare(float1, float2) < 0.

    // Non-zero and non-NaN checking.
    if (float1 < float2) {
      return true;
    }
    if (float1 > float2) {
      return false;
    }
    if (float1 == float2 && float1 != 0.0f) {
      return false;
    }

    // NaNs are equal to other NaNs and larger than any other float
    if (Float.isNaN(float1)) {
      return false;
    }
    if (Float.isNaN(float2)) {
      return true;
    }

    // Deal with +0.0 and -0.0
    int f1 = Float.floatToRawIntBits(float1);
    int f2 = Float.floatToRawIntBits(float2);
    return f1 < f2;
  }
}
