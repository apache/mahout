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

package org.apache.mahout.math.set;

import org.apache.mahout.math.function.${keyTypeCap}Procedure;
import org.apache.mahout.math.list.${keyTypeCap}ArrayList;
import java.util.Arrays;
import java.nio.IntBuffer;

public abstract class Abstract${keyTypeCap}Set extends AbstractSet {

  /**
   * Returns <tt>true</tt> if the receiver contains the specified key.
   *
   * @return <tt>true</tt> if the receiver contains the specified key.
   */
  public boolean contains(final ${keyType} key) {
    return !forEachKey(
        new ${keyTypeCap}Procedure() {
          @Override
          public boolean apply(${keyType} iterKey) {
            return (key != iterKey);
          }
        }
    );
  }

  /**
   * Returns a deep copy of the receiver; uses <code>clone()</code> and casts the result.
   *
   * @return a deep copy of the receiver.
   */
  public Abstract${keyTypeCap}Set copy() {
    return (Abstract${keyTypeCap}Set) clone();
  }

  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }

    if (!(obj instanceof Abstract${keyTypeCap}Set)) {
      return false;
    }
    final Abstract${keyTypeCap}Set other = (Abstract${keyTypeCap}Set) obj;
    if (other.size() != size()) {
      return false;
    }

    return
        forEachKey(
            new ${keyTypeCap}Procedure() {
              @Override
              public boolean apply(${keyType} key) {
                return other.contains(key);
              }
            }
        );
  }

  public int hashCode() {
    final int[] buf = new int[size()];
    forEachKey(
      new ${keyTypeCap}Procedure() {
        int i = 0;

        @Override
        public boolean apply(${keyType} iterKey) {
          buf[i++] = HashUtils.hash(iterKey);
          return true;
        }
      }
    );
    Arrays.sort(buf);
    return IntBuffer.wrap(buf).hashCode();
  }

  /**
   * Applies a procedure to each key of the receiver, if any. Note: Iterates over the keys in no particular order.
   * Subclasses can define a particular order, for example, "sorted by key". All methods which <i>can</i> be expressed
   * in terms of this method (most methods can) <i>must guarantee</i> to use the <i>same</i> order defined by this
   * method, even if it is no particular order. This is necessary so that, for example, methods <tt>keys</tt> and
   * <tt>values</tt> will yield association pairs, not two uncorrelated lists.
   *
   * @param procedure the procedure to be applied. Stops iteration if the procedure returns <tt>false</tt>, otherwise
   *                  continues.
   * @return <tt>false</tt> if the procedure stopped before all keys where iterated over, <tt>true</tt> otherwise.
   */
  public abstract boolean forEachKey(${keyTypeCap}Procedure procedure);

  /**
   * Returns a list filled with all keys contained in the receiver. The returned list has a size that equals
   * <tt>this.size()</tt>. Iteration order is guaranteed to be <i>identical</i> to the order used by method {@link
   * #forEachKey(${keyTypeCap}Procedure)}. <p> This method can be used to iterate over the keys of the receiver.
   *
   * @return the keys.
   */
  public ${keyTypeCap}ArrayList keys() {
    ${keyTypeCap}ArrayList list = new ${keyTypeCap}ArrayList(size());
    keys(list);
    return list;
  }

  /**
   * Fills all keys contained in the receiver into the specified list. Fills the list, starting at index 0. After this
   * call returns the specified list has a new size that equals <tt>this.size()</tt>. Iteration order is guaranteed to
   * be <i>identical</i> to the order used by method {@link #forEachKey(${keyTypeCap}Procedure)}.
   * <p> This method can be used to
   * iterate over the keys of the receiver.
   *
   * @param list the list to be filled, can have any size.
   */
  public void keys(final ${keyTypeCap}ArrayList list) {
    list.clear();
    forEachKey(
        new ${keyTypeCap}Procedure() {
          @Override
          public boolean apply(${keyType} key) {
            list.add(key);
            return true;
          }
        }
    );
  }
  /**
   * Associates the given key with the given value. Replaces any old <tt>(key,someOtherValue)</tt> association, if
   * existing.
   *
   * @param key   the key the value shall be associated with.
   * @return <tt>true</tt> if the receiver did not already contain such a key; <tt>false</tt> if the receiver did
   *         already contain such a key - the new value has now replaced the formerly associated value.
   */
  public abstract boolean add(${keyType} key);

  /**
   * Removes the given key with its associated element from the receiver, if present.
   *
   * @param key the key to be removed from the receiver.
   * @return <tt>true</tt> if the receiver contained the specified key, <tt>false</tt> otherwise.
   */
  public abstract boolean remove(${keyType} key);

  /**
   * Returns a string representation of the receiver, containing the String representation of each key-value pair,
   * sorted ascending by key.
   */
  public String toString() {
    ${keyTypeCap}ArrayList theKeys = keys();
    //theKeys.sort();

    StringBuilder buf = new StringBuilder();
    buf.append('[');
    int maxIndex = theKeys.size() - 1;
    for (int i = 0; i <= maxIndex; i++) {
      ${keyType} key = theKeys.get(i);
      buf.append(String.valueOf(key));
      if (i < maxIndex) {
        buf.append(", ");
      }
    }
    buf.append(']');
    return buf.toString();
  }
}
