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

package org.apache.mahout.graph.linkanalysis;

import com.google.common.primitives.Doubles;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/* models an element of a vector as (index,value) tuple */
public class VectorElementWritable implements Writable {

  private int index;
  private double value;

  public VectorElementWritable() {}

  public VectorElementWritable(int index, double value) {
    this.index = index;
    this.value = value;
  }

  public int index() {
    return index;
  }

  public double get() {
    return value;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(index);
    out.writeDouble(value);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    index = in.readInt();
    value = in.readDouble();
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof VectorElementWritable) {
      VectorElementWritable other = (VectorElementWritable) o;
      return index == other.index && value == other.value;
    }
    return false;
  }

  @Override
  public int hashCode() {
    return index + 31 * Doubles.hashCode(value);
  }
}