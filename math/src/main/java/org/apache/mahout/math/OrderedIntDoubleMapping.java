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

package org.apache.mahout.math;

import java.io.Serializable;

public final class OrderedIntDoubleMapping implements Serializable, Cloneable {

  static final double DEFAULT_VALUE = 0.0;

  private int[] indices;
  private double[] values;
  private int numMappings;

  // If true, doesn't allow DEFAULT_VALUEs in the mapping (adding a zero discards it). Otherwise, a DEFAULT_VALUE is
  // treated like any other value.
  private boolean noDefault = true;

  OrderedIntDoubleMapping(boolean noDefault) {
    this();
    this.noDefault = noDefault;
  }

  OrderedIntDoubleMapping() {
    // no-arg constructor for deserializer
    this(11);
  }

  OrderedIntDoubleMapping(int capacity) {
    indices = new int[capacity];
    values = new double[capacity];
    numMappings = 0;
  }

  OrderedIntDoubleMapping(int[] indices, double[] values, int numMappings) {
    this.indices = indices;
    this.values = values;
    this.numMappings = numMappings;
  }

  public int[] getIndices() {
    return indices;
  }

  public int indexAt(int offset) {
    return indices[offset];
  }

  public void setIndexAt(int offset, int index) {
    indices[offset] = index;
  }

  public double[] getValues() {
    return values;
  }

  public void setValueAt(int offset, double value) {
    values[offset] = value;
  }


  public int getNumMappings() {
    return numMappings;
  }

  private void growTo(int newCapacity) {
    if (newCapacity > indices.length) {
      int[] newIndices = new int[newCapacity];
      System.arraycopy(indices, 0, newIndices, 0, numMappings);
      indices = newIndices;
      double[] newValues = new double[newCapacity];
      System.arraycopy(values, 0, newValues, 0, numMappings);
      values = newValues;
    }
  }

  private int find(int index) {
    int low = 0;
    int high = numMappings - 1;
    while (low <= high) {
      int mid = low + (high - low >>> 1);
      int midVal = indices[mid];
      if (midVal < index) {
        low = mid + 1;
      } else if (midVal > index) {
        high = mid - 1;
      } else {
        return mid;
      }
    }
    return -(low + 1);
  }

  public double get(int index) {
    int offset = find(index);
    return offset >= 0 ? values[offset] : DEFAULT_VALUE;
  }

  public void set(int index, double value) {
    if (numMappings == 0 || index > indices[numMappings - 1]) {
      if (!noDefault || value != DEFAULT_VALUE) {
        if (numMappings >= indices.length) {
          growTo(Math.max((int) (1.2 * numMappings), numMappings + 1));
        }
        indices[numMappings] = index;
        values[numMappings] = value;
        ++numMappings;
      }
    } else {
      int offset = find(index);
      if (offset >= 0) {
        insertOrUpdateValueIfPresent(offset, value);
      } else {
        insertValueIfNotDefault(index, offset, value);
      }
    }
  }

  /**
   * Merges the updates in linear time by allocating new arrays and iterating through the existing indices and values
   * and the updates' indices and values at the same time while selecting the minimum index to set at each step.
   * @param updates another list of mappings to be merged in.
   */
  public void merge(OrderedIntDoubleMapping updates) {
    int[] updateIndices = updates.getIndices();
    double[] updateValues = updates.getValues();

    int newNumMappings = numMappings + updates.getNumMappings();
    int newCapacity = Math.max((int) (1.2 * newNumMappings), newNumMappings + 1);
    int[] newIndices = new int[newCapacity];
    double[] newValues = new double[newCapacity];

    int k = 0;
    int i = 0, j = 0;
    for (; i < numMappings && j < updates.getNumMappings(); ++k) {
      if (indices[i] < updateIndices[j]) {
        newIndices[k] = indices[i];
        newValues[k] = values[i];
        ++i;
      } else if (indices[i] > updateIndices[j]) {
        newIndices[k] = updateIndices[j];
        newValues[k] = updateValues[j];
        ++j;
      } else {
        newIndices[k] = updateIndices[j];
        newValues[k] = updateValues[j];
        ++i;
        ++j;
      }
    }

    for (; i < numMappings; ++i, ++k) {
      newIndices[k] = indices[i];
      newValues[k] = values[i];
    }
    for (; j < updates.getNumMappings(); ++j, ++k) {
      newIndices[k] = updateIndices[j];
      newValues[k] = updateValues[j];
    }

    indices = newIndices;
    values = newValues;
    numMappings = k;
  }

  @Override
  public int hashCode() {
    int result = 0;
    for (int i = 0; i < numMappings; i++) {
      result = 31 * result + indices[i];
      result = 31 * result + (int) Double.doubleToRawLongBits(values[i]);
    }
    return result;
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof OrderedIntDoubleMapping) {
      OrderedIntDoubleMapping other = (OrderedIntDoubleMapping) o;
      if (numMappings == other.numMappings) {
        for (int i = 0; i < numMappings; i++) {
          if (indices[i] != other.indices[i] || values[i] != other.values[i]) {
            return false;
          }
        }
        return true;
      }
    }
    return false;
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder(10 * numMappings);
    for (int i = 0; i < numMappings; i++) {
      result.append('(');
      result.append(indices[i]);
      result.append(',');
      result.append(values[i]);
      result.append(')');
    }
    return result.toString();
  }

  @SuppressWarnings("CloneDoesntCallSuperClone")
  @Override
  public OrderedIntDoubleMapping clone() {
    return new OrderedIntDoubleMapping(indices.clone(), values.clone(), numMappings);
  }

  public void increment(int index, double increment) {
    int offset = find(index);
    if (offset >= 0) {
      double newValue = values[offset] + increment;
      insertOrUpdateValueIfPresent(offset, newValue);
    } else {
      insertValueIfNotDefault(index, offset, increment);
    }
  }

  private void insertValueIfNotDefault(int index, int offset, double value) {
    if (!noDefault || value != DEFAULT_VALUE) {
      if (numMappings >= indices.length) {
        growTo(Math.max((int) (1.2 * numMappings), numMappings + 1));
      }
      int at = -offset - 1;
      if (numMappings > at) {
        for (int i = numMappings - 1, j = numMappings; i >= at; i--, j--) {
          indices[j] = indices[i];
          values[j] = values[i];
        }
      }
      indices[at] = index;
      values[at] = value;
      numMappings++;
    }
  }

  private void insertOrUpdateValueIfPresent(int offset, double newValue) {
    if (noDefault && newValue == DEFAULT_VALUE) {
      for (int i = offset + 1, j = offset; i < numMappings; i++, j++) {
        indices[j] = indices[i];
        values[j] = values[i];
      }
      numMappings--;
    } else {
      values[offset] = newValue;
    }
  }
}
