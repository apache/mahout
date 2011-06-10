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
package org.apache.mahout.clustering;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansClusterer;
import org.apache.mahout.clustering.fuzzykmeans.SoftCluster;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.TimesFunction;

/**
 * This classifier works with any clustering Cluster. It is initialized with a
 * list of compatible clusters and thereafter it can classify any new Vector
 * into one or more of the clusters based upon the pdf() function which each
 * cluster supports.
 * 
 * In addition, it is an OnlineLearner and can be trained. Training amounts to
 * asking the actual model to observe the vector and closing the classifier
 * causes all the models to computeParameters.
 */
public class ClusterClassifier extends AbstractVectorClassifier implements OnlineLearner, Writable {
  
  private List<Cluster> models;
  
  private String modelClass;
  
  /**
   * The public constructor accepts a list of clusters to become the models
   * 
   * @param models
   *          a List<Cluster>
   */
  public ClusterClassifier(List<Cluster> models) {
    this.models = models;
    modelClass = models.get(0).getClass().getName();
  }
  
  // needed for serialization/deserialization
  public ClusterClassifier() {}
  
  @Override
  public Vector classify(Vector instance) {
    Vector pdfs = new DenseVector(models.size());
    if (models.get(0) instanceof SoftCluster) {
      Collection<SoftCluster> clusters = Lists.newArrayList();
      List<Double> distances = Lists.newArrayList();
      for (Cluster model : models) {
        SoftCluster sc = (SoftCluster) model;
        clusters.add(sc);
        distances.add(sc.getMeasure().distance(instance, sc.getCenter()));
      }
      return new FuzzyKMeansClusterer().computePi(clusters, distances);
    } else {
      int i = 0;
      for (Cluster model : models) {
        pdfs.set(i++, model.pdf(new VectorWritable(instance)));
      }
      return pdfs.assign(new TimesFunction(), 1.0 / pdfs.zSum());
    }
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
    for (Cluster cluster : models) {
      cluster.write(out);
    }
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    int size = in.readInt();
    modelClass = in.readUTF();
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    try {
      Class<? extends Cluster> factory = ccl.loadClass(modelClass).asSubclass(
          Cluster.class);
      
      models = Lists.newArrayList();
      for (int i = 0; i < size; i++) {
        Cluster element = factory.newInstance();
        element.readFields(in);
        models.add(element);
      }
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
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
    for (Cluster cluster : models) {
      cluster.computeParameters();
    }
  }
  
  public List<Cluster> getModels() {
    return models;
  }
}
