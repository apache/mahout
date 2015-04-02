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

package org.apache.mahout.clustering.streaming.mapreduce;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class CentroidWritable implements Writable {
  private Centroid centroid = null;

  public CentroidWritable() {}

  public CentroidWritable(Centroid centroid) {
    this.centroid = centroid;
  }

  public Centroid getCentroid() {
    return centroid;
  }

  @Override
  public void write(DataOutput dataOutput) throws IOException {
    dataOutput.writeInt(centroid.getIndex());
    dataOutput.writeDouble(centroid.getWeight());
    VectorWritable.writeVector(dataOutput, centroid.getVector());
  }

  @Override
  public void readFields(DataInput dataInput) throws IOException {
    if (centroid == null) {
      centroid = read(dataInput);
      return;
    }
    centroid.setIndex(dataInput.readInt());
    centroid.setWeight(dataInput.readDouble());
    centroid.assign(VectorWritable.readVector(dataInput));
  }

  public static Centroid read(DataInput dataInput) throws IOException {
    int index = dataInput.readInt();
    double weight = dataInput.readDouble();
    Vector v = VectorWritable.readVector(dataInput);
    return new Centroid(index, v, weight);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof CentroidWritable)) {
      return false;
    }
    CentroidWritable writable = (CentroidWritable) o;
    return centroid.equals(writable.centroid);
  }

  @Override
  public int hashCode() {
    return centroid.hashCode();
  }

  @Override
  public String toString() {
    return centroid.toString();
  }
}
