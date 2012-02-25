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
package org.apache.mahout.clustering.iterator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.TimesFunction;

/**
 * This is a simple maximum likelihood clustering policy, suitable for k-means
 * clustering
 * 
 */
public class MeanShiftClusteringPolicy implements ClusteringPolicy {
  
  public MeanShiftClusteringPolicy() {
    super();
  }
  
  private double t1, t2, t3, t4;
  
  /*
   * (non-Javadoc)
   * 
   * @see
   * org.apache.mahout.clustering.ClusteringPolicy#select(org.apache.mahout.
   * math.Vector)
   */
  @Override
  public Vector select(Vector probabilities) {
    int maxValueIndex = probabilities.maxValueIndex();
    Vector weights = new SequentialAccessSparseVector(probabilities.size());
    weights.set(maxValueIndex, 1.0);
    return weights;
  }
  
  /*
   * (non-Javadoc)
   * 
   * @see
   * org.apache.mahout.clustering.ClusteringPolicy#update(org.apache.mahout.
   * clustering.ClusterClassifier)
   */
  @Override
  public void update(ClusterClassifier posterior) {
    // nothing to do here
  }
  
  @Override
  public Vector classify(Vector data, List<Cluster> models) {
    int i = 0;
    Vector pdfs = new DenseVector(models.size());
    for (Cluster model : models) {
      pdfs.set(i++, model.pdf(new VectorWritable(data)));
    }
    return pdfs.assign(new TimesFunction(), 1.0 / pdfs.zSum());
  }
  
  /*
   * (non-Javadoc)
   * 
   * @see org.apache.hadoop.io.Writable#write(java.io.DataOutput)
   */
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeDouble(t1);
    out.writeDouble(t2);
    out.writeDouble(t3);
    out.writeDouble(t4);
  }
  
  /*
   * (non-Javadoc)
   * 
   * @see org.apache.hadoop.io.Writable#readFields(java.io.DataInput)
   */
  @Override
  public void readFields(DataInput in) throws IOException {
    this.t1 = in.readDouble();
    this.t2 = in.readDouble();
    this.t3 = in.readDouble();
    this.t4 = in.readDouble();
  }
  
}
