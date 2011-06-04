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
package org.apache.mahout.clustering;

import java.io.IOException;
import java.util.Iterator;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * This is an experimental clustering iterator which works with a
 * ClusteringPolicy and a prior ClusterClassifier which has been initialized
 * with a set of models. To date, it has been tested with k-means and Dirichlet
 * clustering. See examples DisplayKMeans and DisplayDirichlet which have been
 * switched over to use it.
 */
public class ClusterIterator {
  
  public ClusterIterator(ClusteringPolicy policy) {
    this.policy = policy;
  }
  
  private final ClusteringPolicy policy;
  
  /**
   * Iterate over data using a prior-trained ClusterClassifier, for a number of
   * iterations
   * 
   * @param data
   *          a {@code List<Vector>} of input vectors
   * @param classifier
   *          a prior ClusterClassifier
   * @param numIterations
   *          the int number of iterations to perform
   * @return the posterior ClusterClassifier
   */
  public ClusterClassifier iterate(Iterable<Vector> data, ClusterClassifier classifier, int numIterations) {
    for (int iteration = 1; iteration <= numIterations; iteration++) {
      for (Vector vector : data) {
        // classification yields probabilities
        Vector probabilities = classifier.classify(vector);
        // policy selects weights for models given those probabilities
        Vector weights = policy.select(probabilities);
        // training causes all models to observe data
        for (Iterator<Vector.Element> it = weights.iterateNonZero(); it.hasNext();) {
          int index = it.next().index();
          classifier.train(index, vector, weights.get(index));
        }
      }
      // compute the posterior models
      classifier.close();
      // update the policy
      policy.update(classifier);
    }
    return classifier;
  }
  
  /**
   * Iterate over data using a prior-trained ClusterClassifier, for a number of
   * iterations
   * 
   * @param inPath
   *          a Path to input VectorWritables
   * @param priorPath
   *          a Path to the prior classifier
   * @param outPath
   *          a Path of output directory
   * @param numIterations
   *          the int number of iterations to perform
   * @throws IOException
   */
  public void iterate(Path inPath, Path priorPath, Path outPath, int numIterations) throws IOException {
    ClusterClassifier classifier = readClassifier(priorPath);
    Configuration conf = new Configuration();
    for (int iteration = 1; iteration <= numIterations; iteration++) {
      for (VectorWritable vw : new SequenceFileDirValueIterable<VectorWritable>(
          inPath, PathType.LIST, PathFilters.logsCRCFilter(), conf)) {
        Vector vector = vw.get();
        // classification yields probabilities
        Vector probabilities = classifier.classify(vector);
        // policy selects weights for models given those probabilities
        Vector weights = policy.select(probabilities);
        // training causes all models to observe data
        for (Iterator<Vector.Element> it = weights.iterateNonZero(); it
            .hasNext();) {
          int index = it.next().index();
          classifier.train(index, vector, weights.get(index));
        }
      }
      // compute the posterior models
      classifier.close();
      // update the policy
      policy.update(classifier);
      // output the classifier
      writeClassifier(classifier, new Path(outPath, "classifier-" + iteration),
          String.valueOf(iteration));
    }
  }
  
  private static void writeClassifier(ClusterClassifier classifier, Path outPath, String k) throws IOException {
    Configuration config = new Configuration();
    FileSystem fs = FileSystem.get(outPath.toUri(), config);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, config, outPath, Text.class, ClusterClassifier.class);
    try {
      Writable key = new Text(k);
      writer.append(key, classifier);
    } finally {
      Closeables.closeQuietly(writer);
    }
  }
  
  private static ClusterClassifier readClassifier(Path inPath) throws IOException {
    Configuration config = new Configuration();
    FileSystem fs = FileSystem.get(inPath.toUri(), config);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, inPath, config);
    Writable key = new Text();
    ClusterClassifier classifierOut = new ClusterClassifier();
    try {
      reader.next(key, classifierOut);
    } finally {
      Closeables.closeQuietly(reader);
    }
    return classifierOut;
  }
}
