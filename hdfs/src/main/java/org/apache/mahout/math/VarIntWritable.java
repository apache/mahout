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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;

public class VarIntWritable implements WritableComparable<VarIntWritable>, Cloneable {

  private int value;

  public VarIntWritable() {
  }

  public VarIntWritable(int value) {
    this.value = value;
  }

  public int get() {
    return value;
  }

  public void set(int value) {
    this.value = value;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof VarIntWritable && ((VarIntWritable) other).value == value;
  }

  @Override
  public int hashCode() {
    return value;
  }

  @Override
  public String toString() {
    return String.valueOf(value);
  }

  @Override
  public VarIntWritable clone() {
    return new VarIntWritable(value);
  }

  @Override
  public int compareTo(VarIntWritable other) {
    if (value < other.value) {
      return -1;
    }
    if (value > other.value) {
      return 1;
    }
    return 0;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeSignedVarInt(value, out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    value = Varint.readSignedVarInt(in);
  }

}
