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
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Varint;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public final class VectorAndPrefsWritable implements Writable {

  private Vector vector;
  private List<Long> userIDs;
  private List<Float> values;

  public VectorAndPrefsWritable() {
  }

  public VectorAndPrefsWritable(Vector vector, List<Long> userIDs, List<Float> values) {
    set(vector, userIDs, values);
  }

  public void set(Vector vector, List<Long> userIDs, List<Float> values) {
    this.vector = vector;
    this.userIDs = userIDs;
    this.values = values;
  }

  public Vector getVector() {
    return vector;
  }

  public List<Long> getUserIDs() {
    return userIDs;
  }

  public List<Float> getValues() {
    return values;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    VectorWritable vw = new VectorWritable(vector);
    vw.setWritesLaxPrecision(true);
    vw.write(out);
    Varint.writeUnsignedVarInt(userIDs.size(), out);
    for (int i = 0; i < userIDs.size(); i++) {
      Varint.writeSignedVarLong(userIDs.get(i), out);
      out.writeFloat(values.get(i));
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    VectorWritable writable = new VectorWritable();
    writable.readFields(in);
    vector = writable.get();
    int size = Varint.readUnsignedVarInt(in);
    userIDs = Lists.newArrayListWithCapacity(size);
    values = Lists.newArrayListWithCapacity(size);
    for (int i = 0; i < size; i++) {
      userIDs.add(Varint.readSignedVarLong(in));
      values.add(in.readFloat());
    }
  }

  @Override
  public String toString() {
    return vector + "\t" + userIDs + '\t' + values;
  }
}
