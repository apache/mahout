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

package org.apache.mahout.common;

import java.io.Serializable;

/** A simple (ordered) pair of two objects. Elements may be null. */
public final class Pair<A,B> implements Comparable<Pair<A,B>>, Serializable {
  
  private final A first;
  private final B second;
  
  public Pair(A first, B second) {
    this.first = first;
    this.second = second;
  }
  
  public A getFirst() {
    return first;
  }
  
  public B getSecond() {
    return second;
  }
  
  public Pair<B, A> swap() {
    return new Pair<B, A>(second, first);
  }

  public static <A,B> Pair<A,B> of(A a, B b) {
    return new Pair<A, B>(a, b);
  }
  
  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof Pair<?, ?>)) {
      return false;
    }
    Pair<?, ?> otherPair = (Pair<?, ?>) obj;
    return isEqualOrNulls(first, otherPair.getFirst())
        && isEqualOrNulls(second, otherPair.getSecond());
  }
  
  private static boolean isEqualOrNulls(Object obj1, Object obj2) {
    return obj1 == null ? obj2 == null : obj1.equals(obj2);
  }
  
  @Override
  public int hashCode() {
    int firstHash = hashCodeNull(first);
    // Flip top and bottom 16 bits; this makes the hash function probably different
    // for (a,b) versus (b,a)
    return (firstHash >>> 16 | firstHash << 16) ^ hashCodeNull(second);
  }
  
  private static int hashCodeNull(Object obj) {
    return obj == null ? 0 : obj.hashCode();
  }
  
  @Override
  public String toString() {
    return '(' + String.valueOf(first) + ',' + second + ')';
  }

  /**
   * Defines an ordering on pairs that sorts by first value's natural ordering, ascending,
   * and then by second value's natural ordering.
   *
   * @throws ClassCastException if types are not actually {@link Comparable}
   */
  @Override
  public int compareTo(Pair<A,B> other) {
    Comparable<A> thisFirst = (Comparable<A>) first;
    A thatFirst = other.getFirst();
    int compare = thisFirst.compareTo(thatFirst);
    if (compare != 0) {
      return compare;
    }
    Comparable<B> thisSecond = (Comparable<B>) second;
    B thatSecond = other.getSecond();
    return thisSecond.compareTo(thatSecond);
  }

}
