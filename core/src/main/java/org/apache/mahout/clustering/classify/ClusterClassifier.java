/* Licensed to the Apache Software Foundation (ASF) under one or more
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
package org.apache.mahout.clustering.classify;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;
import java.util.Locale;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.clustering.iterator.ClusteringPolicyWritable;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

/**
 * This classifier works with any ClusteringPolicy and its associated Clusters.
 * It is initialized with a policy and a list of compatible clusters and
 * thereafter it can classify any new Vector into one or more of the clusters
 * based upon the pdf() function which each cluster supports.
 * 
 * In addition, it is an OnlineLearner and can be trained. Training amounts to
 * asking the actual model to observe the vector and closing the classifier
 * causes all the models to computeParameters.
 * 
 * Because a ClusterClassifier implements Writable, it can be written-to and
 * read-from a sequence file as a single entity. For sequential and mapreduce
 * clustering in conjunction with a ClusterIterator; however, it utilizes an
 * exploded file format. In this format, the iterator writes the policy to a
 * single POLICY_FILE_NAME file in the clustersOut directory and the models are
 * written to one or more part-n files so that multiple reducers may employed to
 * produce them.
 */
public class ClusterClassifier extends AbstractVectorClassifier implements OnlineLearner, Writable {
  
  private static final String POLICY_FILE_NAME = "_policy";
  
  private List<Cluster> models;
  
  private String modelClass;
  
  private ClusteringPolicy policy;
  
  /**
   * The public constructor accepts a list of clusters to become the models
   * 
   * @param models
   *          a List<Cluster>
   * @param policy
   *          a ClusteringPolicy
   */
  public ClusterClassifier(List<Cluster> models, ClusteringPolicy policy) {
    this.models = models;
    modelClass = models.get(0).getClass().getName();
    this.policy = policy;
  }
  
  // needed for serialization/deserialization
  public ClusterClassifier() {}
  
  // only used by MR ClusterIterator
  protected ClusterClassifier(ClusteringPolicy policy) {
    this.policy = policy;
  }
  
  @Override
  public Vector classify(Vector instance) {
    return policy.classify(instance, this);
  }
  
  @Override
  public double classifyScalar(Vector instance) {
    if (models.size() == 2) {
      double pdf0 = models.get(0).pdf(new VectorWritable(instance));
      double pdf1 = models.get(1).pdf(new VectorWritable(instance));
      return pdf0 / (pdf0 + pdf1);
    }
    throw new IllegalStateException();
  }
  
  @Override
  public int numCategories() {
    return models.size();
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(models.size());
    out.writeUTF(modelClass);
    new ClusteringPolicyWritable(policy).write(out);
    for (Cluster cluster : models) {
      cluster.write(out);
    }
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    int size = in.readInt();
    modelClass = in.readUTF();
    models = Lists.newArrayList();
    ClusteringPolicyWritable clusteringPolicyWritable = new ClusteringPolicyWritable();
    clusteringPolicyWritable.readFields(in);
    policy = clusteringPolicyWritable.getValue();
    for (int i = 0; i < size; i++) {
      Cluster element = ClassUtils.instantiateAs(modelClass, Cluster.class);
      element.readFields(in);
      models.add(element);
    }
  }
  
  @Override
  public void train(int actual, Vector instance) {
    models.get(actual).observe(new VectorWritable(instance));
  }
  
  /**
   * Train the models given an additional weight. Unique to ClusterClassifier
   * 
   * @param actual
   *          the int index of a model
   * @param data
   *          a data Vector
   * @param weight
   *          a double weighting factor
   */
  public void train(int actual, Vector data, double weight) {
    models.get(actual).observe(new VectorWritable(data), weight);
  }
  
  @Override
  public void train(long trackingKey, String groupKey, int actual, Vector instance) {
    models.get(actual).observe(new VectorWritable(instance));
  }
  
  @Override
  public void train(long trackingKey, int actual, Vector instance) {
    models.get(actual).observe(new VectorWritable(instance));
  }
  
  @Override
  public void close() {
    policy.close(this);
  }
  
  public List<Cluster> getModels() {
    return models;
  }
  
  public ClusteringPolicy getPolicy() {
    return policy;
  }
  
  public void writeToSeqFiles(Path path) throws IOException {
    writePolicy(policy, path);
    Configuration config = new Configuration();
    FileSystem fs = FileSystem.get(path.toUri(), config);
    SequenceFile.Writer writer = null;
    ClusterWritable cw = new ClusterWritable();
    for (int i = 0; i < models.size(); i++) {
      try {
        Cluster cluster = models.get(i);
        cw.setValue(cluster);
        writer = new SequenceFile.Writer(fs, config,
            new Path(path, "part-" + String.format(Locale.ENGLISH, "%05d", i)), IntWritable.class,
            ClusterWritable.class);
        Writable key = new IntWritable(i);
        writer.append(key, cw);
      } finally {
        Closeables.close(writer, false);
      }
    }
  }
  
  public void readFromSeqFiles(Configuration conf, Path path) throws IOException {
    Configuration config = new Configuration();
    List<Cluster> clusters = Lists.newArrayList();
    for (ClusterWritable cw : new SequenceFileDirValueIterable<ClusterWritable>(path, PathType.LIST,
        PathFilters.logsCRCFilter(), config)) {
      Cluster cluster = cw.getValue();
      cluster.configure(conf);
      clusters.add(cluster);
    }
    this.models = clusters;
    modelClass = models.get(0).getClass().getName();
    this.policy = readPolicy(path);
  }
  
  public static ClusteringPolicy readPolicy(Path path) throws IOException {
    Path policyPath = new Path(path, POLICY_FILE_NAME);
    Configuration config = new Configuration();
    FileSystem fs = FileSystem.get(policyPath.toUri(), config);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, policyPath, config);
    Text key = new Text();
    ClusteringPolicyWritable cpw = new ClusteringPolicyWritable();
    reader.next(key, cpw);
    return cpw.getValue();
  }
  
  public static void writePolicy(ClusteringPolicy policy, Path path) throws IOException {
    Path policyPath = new Path(path, POLICY_FILE_NAME);
    Configuration config = new Configuration();
    FileSystem fs = FileSystem.get(policyPath.toUri(), config);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, config, policyPath, Text.class,
        ClusteringPolicyWritable.class);
    writer.append(new Text(), new ClusteringPolicyWritable(policy));
    writer.close();
  }
}
