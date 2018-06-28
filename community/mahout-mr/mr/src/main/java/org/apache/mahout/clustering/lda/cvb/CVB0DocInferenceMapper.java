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
package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;

public class CVB0DocInferenceMapper extends CachingCVB0Mapper {

  private final VectorWritable topics = new VectorWritable();

  @Override
  public void map(IntWritable docId, VectorWritable doc, Context context)
    throws IOException, InterruptedException {
    int numTopics = getNumTopics();
    Vector docTopics = new DenseVector(numTopics).assign(1.0 / numTopics);
    Matrix docModel = new SparseRowMatrix(numTopics, doc.get().size());
    int maxIters = getMaxIters();
    ModelTrainer modelTrainer = getModelTrainer();
    for (int i = 0; i < maxIters; i++) {
      modelTrainer.getReadModel().trainDocTopicModel(doc.get(), docTopics, docModel);
    }
    topics.set(docTopics);
    context.write(docId, topics);
  }

  @Override
  protected void cleanup(Context context) {
    getModelTrainer().stop();
  }
}
