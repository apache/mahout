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

package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Varint;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class VectorWithIndexWritable implements Writable {

  private Vector vector;
  private Integer idIndex;

  public VectorWithIndexWritable() {
  }

  public VectorWithIndexWritable(Vector vector) {
    this.vector = vector;
  }

  public VectorWithIndexWritable(int idIndex) {
    this.idIndex = idIndex;
  }

  public VectorWithIndexWritable(Integer idIndex, Vector vector) {
    this.vector = vector;
    this.idIndex = idIndex;
  }

  public Vector getVector() {
    return vector;
  }

  public int getIDIndex() {
    return idIndex;
  }

  public boolean hasVector() {
    return vector != null;
  }

  public boolean hasIndex() {
    return idIndex != null;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(hasVector());
    if (hasVector()) {
      new VectorWritable(vector).write(out);
    }
    out.writeBoolean(hasIndex());
    if (hasIndex()) {
      Varint.writeSignedVarInt(idIndex, out);
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    vector = null;
    idIndex = null;
    boolean hasVector = in.readBoolean();
    if (hasVector) {
      VectorWritable writable = new VectorWritable();
      writable.readFields(in);
      vector = writable.get();
    }
    boolean hasRating = in.readBoolean();
    if (hasRating) {
      idIndex = Varint.readSignedVarInt(in);
    }
  }
}
