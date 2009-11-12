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

package org.apache.mahout.classifier.bayes.datastore;

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.bayes.exceptions.InvalidDatastoreException;
import org.apache.mahout.classifier.bayes.interfaces.Datastore;
import org.apache.mahout.classifier.bayes.io.SequenceFileModelReader;
import org.apache.mahout.common.Parameters;

public class InMemoryBayesDatastore implements Datastore {

  private final Map<String, Map<String, Map<String, Double>>> matrices = new HashMap<String, Map<String, Map<String, Double>>>();

  private final Map<String, Map<String, Double>> vectors = new HashMap<String, Map<String, Double>>();

  private Parameters params = null;

  private double thetaNormalizer = 1.0;

  private double alpha_i = 1.0;

  public InMemoryBayesDatastore(Parameters params) {

    matrices.put("weight", new HashMap<String, Map<String, Double>>());
    vectors.put("sumWeight", new HashMap<String, Double>());
    matrices.put("weight", new HashMap<String, Map<String, Double>>());
    vectors.put("labelWeight", new HashMap<String, Double>());
    vectors.put("thetaNormalizer", new HashMap<String, Double>());
    String basePath = params.get("basePath");
    this.params = params;
    params.set("sigma_j", basePath + "/trainer-weights/Sigma_j/part-*");
    params.set("sigma_k", basePath + "/trainer-weights/Sigma_k/part-*");
    params.set("sigma_kSigma_j", basePath
        + "/trainer-weights/Sigma_kSigma_j/part-*");
    params.set("thetaNormalizer", basePath + "/trainer-thetaNormalizer/part-*");
    params.set("weight", basePath + "/trainer-tfIdf/trainer-tfIdf/part-*");
    alpha_i = Double.valueOf(params.get("alpha_i", "1.0"));
  }

  @Override
  public void initialize() throws InvalidDatastoreException {
    Configuration conf = new Configuration();
    String basePath = params.get("basePath");
    try {
      SequenceFileModelReader.loadModel(this, FileSystem.get(new Path(basePath)
          .toUri(), conf), params, conf);
    } catch (IOException e) {
      throw new InvalidDatastoreException(e.getMessage());
    }
    updateVocabCount();
    Collection<String> labels = getKeys("thetaNormalizer");
    for (String label : labels) {
      thetaNormalizer = Math.max(thetaNormalizer, Math.abs(vectorGetCell(
          "thetaNormalizer", label)));
    }
    for (String label : labels) {
      System.out.println(label + ' ' + vectorGetCell("thetaNormalizer", label)
          + ' ' + thetaNormalizer + ' '
          + vectorGetCell("thetaNormalizer", label) / thetaNormalizer);
    }
  }

  @Override
  public Collection<String> getKeys(String name)
      throws InvalidDatastoreException {
    return vectors.get("labelWeight").keySet();
  }

  @Override
  public double getWeight(String matrixName, String row, String column)
      throws InvalidDatastoreException {
    return matrixGetCell(matrixName, row, column);
  }

  @Override
  public double getWeight(String vectorName, String index)
      throws InvalidDatastoreException {
    if (vectorName.equals("thetaNormalizer"))
      return vectorGetCell(vectorName, index) / thetaNormalizer;
    else if (vectorName.equals("params")) {
      if(index.equals("alpha_i")) return alpha_i;
      else throw new InvalidDatastoreException();
    } 
    return vectorGetCell(vectorName, index);
  }

  private double matrixGetCell(String matrixName, String row, String col)
      throws InvalidDatastoreException {
    Map<String, Map<String, Double>> matrix = matrices.get(matrixName);
    if (matrix == null) {
      throw new InvalidDatastoreException();
    }
    Map<String, Double> rowVector = matrix.get(row);
    if (rowVector == null) {
      return 0.0;
    }
    return nullToZero(rowVector.get(col));
  }

  private double vectorGetCell(String vectorName, String index)
      throws InvalidDatastoreException {

    Map<String, Double> vector = vectors.get(vectorName);
    if (vector == null) {
      throw new InvalidDatastoreException();
    }
    return nullToZero(vector.get(index));
  }

  private void matrixPutCell(String matrixName, String row, String col,
      double weight) {
    Map<String, Map<String, Double>> matrix = matrices.get(matrixName);
    if (matrix == null) {
      matrix = new HashMap<String, Map<String, Double>>();
      matrices.put(matrixName, matrix);
    }
    Map<String, Double> rowVector = matrix.get(row);
    if (rowVector == null) {
      rowVector = new HashMap<String, Double>();
      matrix.put(row, rowVector);
    }
    rowVector.put(col, weight);
  }

  private void vectorPutCell(String vectorName, String index, double weight) {

    Map<String, Double> vector = vectors.get(vectorName);
    if (vector == null) {
      vector = new HashMap<String, Double>();
      vectors.put(vectorName, vector);
    }
    vector.put(index, weight);
  }

  private long sizeOfMatrix(String matrixName) {
    Map<String, Map<String, Double>> matrix = matrices.get(matrixName);
    if (matrix == null) {
      return 0;
    }
    return matrix.size();
  }

  public void loadFeatureWeight(String feature, String label, double weight) {
    matrixPutCell("weight", feature, label, weight);
  }

  public void setSumFeatureWeight(String feature, double weight) {
    matrixPutCell("weight", feature, "sigma_j", weight);
  }

  public void setSumLabelWeight(String label, double weight) {
    vectorPutCell("labelWeight", label, weight);
  }

  public void setThetaNormalizer(String label, double weight) {
    vectorPutCell("thetaNormalizer", label, weight);
  }

  public void setSigma_jSigma_k(double weight) {
    vectorPutCell("sumWeight", "sigma_jSigma_k", weight);
  }

  public void updateVocabCount() {
    vectorPutCell("sumWeight", "vocabCount", sizeOfMatrix("weight"));
  }

  private static double nullToZero(Double value) {
    return value == null ? 0.0 : value;
  }

}
