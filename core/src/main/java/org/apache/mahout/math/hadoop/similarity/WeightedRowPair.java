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

package org.apache.mahout.math.hadoop.similarity;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.math.Varint;

/**
 * a pair of row vectors that has at least one entry != NaN in the same column together with the precomputed weights of
 * the row vectors
 */
public class WeightedRowPair implements WritableComparable<WeightedRowPair> {

  private int rowA;
  private int rowB;
  private double weightA;
  private double weightB;

  public WeightedRowPair() {
  }

  public WeightedRowPair(int rowA, int rowB, double weightA, double weightB) {
    this.rowA = rowA;
    this.rowB = rowB;
    this.weightA = weightA;
    this.weightB = weightB;
  }

  public void set(int rowA, int rowB, double weightA, double weightB) {
    this.rowA = rowA;
    this.rowB = rowB;
    this.weightA = weightA;
    this.weightB = weightB;
  }

  public int getRowA() {
    return rowA;
  }
  public int getRowB() {
    return rowB;
  }
  public double getWeightA() {
    return weightA;
  }
  public double getWeightB() {
    return weightB;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    rowA = Varint.readSignedVarInt(in);
    rowB = Varint.readSignedVarInt(in);
    weightA = in.readDouble();
    weightB = in.readDouble();
  }
  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeSignedVarInt(rowA, out);
    Varint.writeSignedVarInt(rowB, out);
    out.writeDouble(weightA);
    out.writeDouble(weightB);
  }
  @Override
  public int compareTo(WeightedRowPair other) {
    int result = compare(rowA, other.rowA);
    if (result == 0) {
      result = compare(rowB, other.rowB);
    }
    return result;
  }

  @Override
  public int hashCode() {
    return rowA + 31 * rowB;
  }

  @Override
  public boolean equals(Object other) {
    if (other instanceof WeightedRowPair) {
      WeightedRowPair otherPair = (WeightedRowPair) other;
      return rowA == otherPair.rowA && rowB == otherPair.rowB;
    }
    return false;
  }

  protected static int compare(int a, int b) {
    return a == b ? 0 : a < b ? -1 : 1;
  }
}
