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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Varint;

public class SimilarItemWritable implements Writable {

  private long itemID;
  private double value;

  public SimilarItemWritable() {
    super();
  }

  public SimilarItemWritable(long itemID, double value) {
    super();
    this.itemID = itemID;
    this.value = value;
  }

  public long getItemID() {
    return itemID;
  }

  public double getValue() {
    return value;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    itemID = Varint.readSignedVarLong(in);
    value = in.readDouble();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeSignedVarLong(itemID, out);
    out.writeDouble(value);
  }

  @Override
  public int hashCode() {
    return RandomUtils.hashLong(itemID);
  }

  @Override
  public boolean equals(Object other) {
    if (other instanceof SimilarItemWritable) {
      return (itemID == ((SimilarItemWritable)other).itemID);
    }
    return false;
  }
}
