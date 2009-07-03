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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class FuzzyKMeansOutput implements Writable {
  //parallel arrays
  private SoftCluster[] clusters;
  private double[] probabilities;

  public FuzzyKMeansOutput() {
  }

  public FuzzyKMeansOutput(int size) {
    clusters = new SoftCluster[size];
    probabilities = new double[size];
  }

  public SoftCluster[] getClusters() {
    return clusters;
  }

  public double[] getProbabilities() {
    return probabilities;
  }

  public void add(int i, SoftCluster softCluster, double probWeight) {
    clusters[i] = softCluster;
    probabilities[i] = probWeight;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(clusters.length);
    for (SoftCluster cluster : clusters) {
      cluster.write(out);
    }
    out.writeInt(probabilities.length);
    for (double probability : probabilities) {
      out.writeDouble(probability);
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int numClusters = in.readInt();
    clusters = new SoftCluster[numClusters];
    for (int i = 0; i < numClusters; i++) {
      clusters[i] = new SoftCluster();
      clusters[i].readFields(in);
    }
    int numProbs = in.readInt();
    probabilities = new double[numProbs];
    for (int i = 0; i < numProbs; i++) {
      probabilities[i] = in.readDouble();
    }
  }
}
