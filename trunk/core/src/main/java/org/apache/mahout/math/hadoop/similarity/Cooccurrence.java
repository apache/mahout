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

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Varint;

/**
 * a pair of entries in the same column of a row vector where each of the entries' values is != NaN
 */
public class Cooccurrence implements Writable {

  private int column;
  private double valueA;
  private double valueB;

  public Cooccurrence() {
  }

  public Cooccurrence(int column, double valueA, double valueB) {
    this.column = column;
    this.valueA = valueA;
    this.valueB = valueB;
  }

  public void set(int column, double valueA, double valueB) {
    this.column = column;
    this.valueA = valueA;
    this.valueB = valueB;
  }

  public int getColumn() {
    return column;
  }
  public double getValueA() {
    return valueA;
  }
  public double getValueB() {
    return valueB;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    column = Varint.readSignedVarInt(in);
    valueA = in.readDouble();
    valueB = in.readDouble();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeSignedVarInt(column, out);
    out.writeDouble(valueA);
    out.writeDouble(valueB);
  }

  @Override
  public int hashCode() {
    return column;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof Cooccurrence && column == ((Cooccurrence) other).column;
  }
}
