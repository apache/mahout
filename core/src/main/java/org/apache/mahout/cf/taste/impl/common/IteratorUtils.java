/**
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

package org.apache.mahout.cf.taste.impl.common;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/** <p>{@link java.util.Iterator}-related methods without a better home.</p> */
public final class IteratorUtils {

  private IteratorUtils() {
  }

  /**
   * @param iterable {@link Iterable} whose contents are to be put into a {@link List}
   * @return a {@link List} with the objects one gets by iterating over the given {@link Iterable}
   */
  public static <K> List<K> iterableToList(Iterable<K> iterable) {
    return iterableToList(iterable, null);
  }

  public static long[] longIteratorToList(LongPrimitiveIterator iterator) {
    long[] result = new long[5];
    int size = 0;
    while (iterator.hasNext()) {
      if (size == result.length) {
        long[] newResult = new long[(result.length << 1)];
        System.arraycopy(result, 0, newResult, 0, result.length);
        result = newResult;
      }
      result[size++] = iterator.next();
    }
    if (size != result.length) {
      long[] newResult = new long[size];
      System.arraycopy(result, 0, newResult, 0, size);
      result = newResult;
    }
    return result;
  }

  /**
   * @param iterable   {@link Iterable} whose contents are to be put into a {@link List}
   * @param comparator {@link Comparator} defining the sort order of the returned {@link List}
   * @return a {@link List} with the objects one gets by iterating over the given {@link Iterable}, sorted according to
   *         the given {@link Comparator}
   */
  public static <K> List<K> iterableToList(Iterable<K> iterable, Comparator<K> comparator) {
    if (iterable == null) {
      throw new IllegalArgumentException("iterable is null");
    }
    List<K> list;
    if (iterable instanceof Collection) {
      if (iterable instanceof List) {
        list = (List<K>) iterable;
      } else {
        Collection<K> collection = (Collection<K>) iterable;
        list = new ArrayList<K>(collection.size());
        list.addAll(collection);
      }
    } else {
      list = new ArrayList<K>();
      for (K item : iterable) {
        list.add(item);
      }
    }
    if (comparator != null) {
      Collections.sort(list, comparator);
    }
    return list;
  }

}
