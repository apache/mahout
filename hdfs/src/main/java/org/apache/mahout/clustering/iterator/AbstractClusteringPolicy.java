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

public abstract class AbstractClusteringPolicy implements ClusteringPolicy {
  
  @Override
  public abstract void write(DataOutput out) throws IOException;
  
  @Override
  public abstract void readFields(DataInput in) throws IOException;
  
  @Override
  public Vector select(Vector probabilities) {
    int maxValueIndex = probabilities.maxValueIndex();
    Vector weights = new SequentialAccessSparseVector(probabilities.size());
    weights.set(maxValueIndex, 1.0);
    return weights;
  }
  
  @Override
  public void update(ClusterClassifier posterior) {
    // nothing to do in general here
  }
  
  @Override
  public Vector classify(Vector data, ClusterClassifier prior) {
    List<Cluster> models = prior.getModels();
    int i = 0;
    Vector pdfs = new DenseVector(models.size());
    for (Cluster model : models) {
      pdfs.set(i++, model.pdf(new VectorWritable(data)));
    }
    return pdfs.assign(new TimesFunction(), 1.0 / pdfs.zSum());
  }
  
  @Override
  public void close(ClusterClassifier posterior) {
    for (Cluster cluster : posterior.getModels()) {
      cluster.computeParameters();
    }
    
  }
  
}
