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

package org.apache.mahout.clustering.fuzzykmeans;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.Vector;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class FuzzyKMeansInfo implements Writable {

  private double probability;
  private Vector pointTotal;

  public int combinerPass = 0;

  public FuzzyKMeansInfo() {
  }

  public FuzzyKMeansInfo(double probability, Vector pointTotal) {
    this.probability = probability;
    this.pointTotal = pointTotal;
  }

  public FuzzyKMeansInfo(double probability, Vector pointTotal, int combinerPass) {
    this.probability = probability;
    this.pointTotal = pointTotal;
    this.combinerPass = combinerPass;
  }

  public Vector getVector() {
    return pointTotal;
  }

  public double getProbability() {
    return probability;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeDouble(probability);
    AbstractVector.writeVector(out, pointTotal);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.probability = in.readDouble();
    this.pointTotal = AbstractVector.readVector(in);
  }
}