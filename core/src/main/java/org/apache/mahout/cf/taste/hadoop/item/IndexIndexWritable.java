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

package org.apache.mahout.cf.taste.hadoop.item;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableComparable;
import org.apache.mahout.math.Varint;

/** A {@link WritableComparable} encapsulating two item indices. */
public final class IndexIndexWritable
    implements WritableComparable<IndexIndexWritable>, Cloneable {

  private int aID;
  private int bID;

  public IndexIndexWritable() {
  // do nothing
  }

  public IndexIndexWritable(int aID, int bID) {
    this.aID = aID;
    this.bID = bID;
  }

  public int getAID() {
    return aID;
  }

  public int getBID() {
    return bID;
  }

  public void set(int aID, int bID) {
    this.aID = aID;
    this.bID = bID;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeUnsignedVarInt(aID, out);
    Varint.writeUnsignedVarInt(bID, out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    aID = Varint.readUnsignedVarInt(in);
    bID = Varint.readUnsignedVarInt(in);    
  }

  @Override
  public int compareTo(IndexIndexWritable that) {
    int aCompare = compare(aID, that.getAID());
    return aCompare == 0 ? compare(bID, that.getBID()) : aCompare;
  }

  private static int compare(int a, int b) {
    return a < b ? -1 : a > b ? 1 : 0;
  }

  @Override
  public int hashCode() {
    return aID + 31 * bID;
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof IndexIndexWritable) {
      IndexIndexWritable that = (IndexIndexWritable) o;
      return aID == that.getAID() && bID == that.getBID();
    }
    return false;
  }

  @Override
  public String toString() {
    return aID + "\t" + bID;
  }

  @Override
  public IndexIndexWritable clone() {
    return new IndexIndexWritable(aID, bID);
  }

}