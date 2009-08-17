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

package org.apache.mahout.clustering.lda;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

/**
* Saves two ints, x and y.
*/
public class IntPairWritable implements WritableComparable<IntPairWritable> {

  private int x;
  private int y;

  /** For serialization purposes only */
  public IntPairWritable() {
  }

  public IntPairWritable(int x, int y) {
    this.x = x;
    this.y = y;
  }

  public void setX(int x) {
    this.x = x;
  }

  public int getX() {
    return x;
  }

  public void setY(int y) {
    this.y = y;
  }

  public int getY() {
    return y;
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeInt(x);
    dataOutput.writeInt(y);
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    x = dataInput.readInt();
    y = dataInput.readInt();
  }

  public int compareTo(IntPairWritable that) {
    int xdiff = this.x - that.x;
    return (xdiff != 0) ? xdiff : this.y - that.y;
  }

  public boolean equals(Object o) {
    if (this == o) { 
      return true;
    } else if (!(o instanceof IntPairWritable)) {
      return false;
    }

    IntPairWritable that = (IntPairWritable) o;

    return that.x == this.x && this.y == that.y;
  }

  @Override
  public int hashCode() {
    return 43 * x + y;
  }

  @Override
  public String toString() {
    return "(" + x + ", " + y + ")";
  }

  static {
    WritableComparator.define(IntPairWritable.class, new Comparator());
  }

  public static class Comparator extends WritableComparator {
    public Comparator() {
      super(IntPairWritable.class);
    }

    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      assert l1 == 8;
      int int11 = readInt(b1, s1);
      int int21 = readInt(b2, s2);
      if (int11 != int21) {
        return int11 - int21;
      }

      int int12 = readInt(b1, s1 + 4);
      int int22 = readInt(b2, s2 + 4);
      return int12 - int22;
    }
  }
}
