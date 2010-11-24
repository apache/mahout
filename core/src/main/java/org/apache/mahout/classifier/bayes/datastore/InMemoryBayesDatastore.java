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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.bayes.exceptions.InvalidDatastoreException;
import org.apache.mahout.classifier.bayes.interfaces.Datastore;
import org.apache.mahout.classifier.bayes.io.SequenceFileModelReader;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;
import org.apache.mahout.math.map.OpenObjectIntHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Class implementing the Datastore for Algorithms to read In-Memory model
 * 
 */
public class InMemoryBayesDatastore implements Datastore {
  
  private static final Logger log = LoggerFactory.getLogger(InMemoryBayesDatastore.class);
  
  private final OpenObjectIntHashMap<String> featureDictionary = new OpenObjectIntHashMap<String>();
  
  private final OpenObjectIntHashMap<String> labelDictionary = new OpenObjectIntHashMap<String>();
  
  private final OpenIntDoubleHashMap sigmaJ = new OpenIntDoubleHashMap();
  
  private final OpenIntDoubleHashMap sigmaK = new OpenIntDoubleHashMap();
  
  private final OpenIntDoubleHashMap thetaNormalizerPerLabel = new OpenIntDoubleHashMap();
  
  private final Matrix weightMatrix = new SparseMatrix(new int[] {1, 0});
  
  private final Parameters params;
  
  private double thetaNormalizer = 1.0;
  
  private double alphaI = 1.0;
  
  private double sigmaJsigmaK = 1.0;
  
  public InMemoryBayesDatastore(Parameters params) {
    String basePath = params.get("basePath");
    this.params = params;
    params.set("sigma_j", basePath + "/trainer-weights/Sigma_j/part-*");
    params.set("sigma_k", basePath + "/trainer-weights/Sigma_k/part-*");
    params.set("sigma_kSigma_j", basePath + "/trainer-weights/Sigma_kSigma_j/part-*");
    params.set("thetaNormalizer", basePath + "/trainer-thetaNormalizer/part-*");
    params.set("weight", basePath + "/trainer-tfIdf/trainer-tfIdf/part-*");
    alphaI = Double.valueOf(params.get("alpha_i", "1.0"));
  }
  
  @Override
  public void initialize() throws InvalidDatastoreException {
    Configuration conf = new Configuration();
    String basePath = params.get("basePath");
    try {
      SequenceFileModelReader.loadModel(this, FileSystem.get(new Path(basePath).toUri(), conf), params, conf);
    } catch (IOException e) {
      throw new InvalidDatastoreException(e);
    }
    for (String label : getKeys("")) {
      log.info("{} {} {} {}", new Object[] {
        label,
        thetaNormalizerPerLabel.get(getLabelID(label)),
        thetaNormalizer,
        thetaNormalizerPerLabel.get(getLabelID(label)) / thetaNormalizer
      });
    }
  }
  
  @Override
  public Collection<String> getKeys(String name) throws InvalidDatastoreException {
    return labelDictionary.keys();
  }
  
  @Override
  public double getWeight(String matrixName, String row, String column) throws InvalidDatastoreException {
    if ("weight".equals(matrixName)) {
      if ("sigma_j".equals(column)) {
        return sigmaJ.get(getFeatureID(row));
      } else {
        return weightMatrix.getQuick(getFeatureID(row), getLabelID(column));
      }
    } else {
      throw new InvalidDatastoreException("Matrix not found: " + matrixName);
    }
  }
  
  @Override
  public double getWeight(String vectorName, String index) throws InvalidDatastoreException {
    if ("sumWeight".equals(vectorName)) {
      if ("sigma_jSigma_k".equals(index)) {
        return sigmaJsigmaK;
      } else if ("vocabCount".equals(index)) {
        return featureDictionary.size();
      } else {
        throw new InvalidDatastoreException();
      }
    } else if ("thetaNormalizer".equals(vectorName)) {
      return thetaNormalizerPerLabel.get(getLabelID(index)) / thetaNormalizer;
    } else if ("params".equals(vectorName)) {
      if ("alpha_i".equals(index)) {
        return alphaI;
      } else {
        throw new InvalidDatastoreException();
      }
    } else if ("labelWeight".equals(vectorName)) {
      return sigmaK.get(getLabelID(index));
    } else {
      throw new InvalidDatastoreException();
    }
  }
  
  private int getFeatureID(String feature) {
    if (featureDictionary.containsKey(feature)) {
      return featureDictionary.get(feature);
    } else {
      int id = featureDictionary.size();
      featureDictionary.put(feature, id);
      return id;
    }
  }
  
  private int getLabelID(String label) {
    if (labelDictionary.containsKey(label)) {
      return labelDictionary.get(label);
    } else {
      int id = labelDictionary.size();
      labelDictionary.put(label, id);
      return id;
    }
  }
  
  public void loadFeatureWeight(String feature, String label, double weight) {
    int fid = getFeatureID(feature);
    int lid = getLabelID(label);
    weightMatrix.setQuick(fid, lid, weight);
  }
  
  public void setSumFeatureWeight(String feature, double weight) {
    int fid = getFeatureID(feature);
    sigmaJ.put(fid, weight);
  }
  
  public void setSumLabelWeight(String label, double weight) {
    int lid = getLabelID(label);
    sigmaK.put(lid, weight);
  }
  
  public void setThetaNormalizer(String label, double weight) {
    int lid = getLabelID(label);
    thetaNormalizerPerLabel.put(lid, weight);
    thetaNormalizer = Math.max(thetaNormalizer, Math.abs(weight));
  }
  
  public void setSigmaJSigmaK(double weight) {
    this.sigmaJsigmaK = weight;
  }
}
