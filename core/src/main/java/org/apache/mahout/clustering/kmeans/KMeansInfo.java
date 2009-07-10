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

package org.apache.mahout.clustering.kmeans;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.Vector;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class KMeansInfo implements Writable {

  private int points;
  private Vector pointTotal;

  public KMeansInfo() {
  }

  public KMeansInfo(int points, Vector pointTotal) {
    this.points = points;
    this.pointTotal = pointTotal;
  }

  public int getPoints() {
    return points;
  }

  public Vector getPointTotal() {
    return pointTotal;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(points);
    AbstractVector.writeVector(out, pointTotal);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.points = in.readInt();
    this.pointTotal = AbstractVector.readVector(in);
  }
}
