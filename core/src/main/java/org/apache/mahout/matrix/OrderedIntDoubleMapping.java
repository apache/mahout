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

package org.apache.mahout.matrix;

import java.io.Serializable;

final class OrderedIntDoubleMapping implements Serializable, Cloneable {

  private static final double DEFAULT_VALUE = 0.0;

  private int[] indices;
  private double[] values;
  private int numMappings;

  OrderedIntDoubleMapping() {
    // no-arg constructor for deserializer
    this(11);
  }

  OrderedIntDoubleMapping(int capacity) {
    indices = new int[capacity];
    values = new double[capacity];
    numMappings = 0;
  }

  int[] getIndices() {
    return indices;
  }

  double[] getValues() {
    return values;
  }

  int getNumMappings() {
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
      int mid = low + ((high - low) >>> 1);
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
    int offset = find(index);
    if (offset >= 0) {
      if (value == DEFAULT_VALUE) {
        for (int i = offset + 1, j = offset; i < numMappings; i++, j++) {
          indices[j] = indices[i];
          values[j] = values[i];
        }
        numMappings--;
      } else {
        values[offset] = value;
      }
    } else {
      if (value != DEFAULT_VALUE) {
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

  @Override
  public OrderedIntDoubleMapping clone() {
    OrderedIntDoubleMapping clone = new OrderedIntDoubleMapping(numMappings);
    System.arraycopy(indices, 0, clone.indices, 0, numMappings);
    System.arraycopy(values, 0, clone.values, 0, numMappings);
    clone.numMappings = numMappings;
    return clone;
  }

}
