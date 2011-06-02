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

package org.apache.mahout.classifier.naivebayes;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.naivebayes.trainer.NaiveBayesTrainer;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * NaiveBayesModel holds the weight Matrix, the feature and label sums and the weight normalizer vectors.
 */
public class NaiveBayesModel {

  private static final String MODEL = "NaiveBayesModel";

  private Vector labelSum;
  private Vector perlabelThetaNormalizer;
  private Vector featureSum;
  private Matrix weightMatrix;
  private float alphaI;
  private double vocabCount;
  private double totalSum;
  
  private NaiveBayesModel() { 
    // do nothing
  }
  
  public NaiveBayesModel(Matrix matrix, Vector featureSum, Vector labelSum, Vector thetaNormalizer, float alphaI) {
    this.weightMatrix = matrix;
    this.featureSum = featureSum;
    this.labelSum = labelSum;
    this.perlabelThetaNormalizer = thetaNormalizer;
    this.vocabCount = featureSum.getNumNondefaultElements();
    this.totalSum = labelSum.zSum();
    this.alphaI = alphaI;
  }

  private void setLabelSum(Vector labelSum) {
    this.labelSum = labelSum;
  }


  public void setPerlabelThetaNormalizer(Vector perlabelThetaNormalizer) {
    this.perlabelThetaNormalizer = perlabelThetaNormalizer;
  }


  public void setFeatureSum(Vector featureSum) {
    this.featureSum = featureSum;
  }


  public void setWeightMatrix(Matrix weightMatrix) {
    this.weightMatrix = weightMatrix;
  }


  public void setAlphaI(float alphaI) {
    this.alphaI = alphaI;
  }


  public void setVocabCount(double vocabCount) {
    this.vocabCount = vocabCount;
  }


  public void setTotalSum(double totalSum) {
    this.totalSum = totalSum;
  }
  
  public Vector getLabelSum() {
    return labelSum;
  }

  public Vector getPerlabelThetaNormalizer() {
    return perlabelThetaNormalizer;
  }

  public Vector getFeatureSum() {
    return featureSum;
  }

  public Matrix getWeightMatrix() {
    return weightMatrix;
  }

  public float getAlphaI() {
    return alphaI;
  }

  public double getVocabCount() {
    return vocabCount;
  }

  public double getTotalSum() {
    return totalSum;
  }
  
  public int getNumLabels() {
    return labelSum.size();
  }

  public static String getModelName() {
    return MODEL;
  }
  
  // CODE USED FOR SERIALIZATION
  public static NaiveBayesModel fromMRTrainerOutput(Path output, Configuration conf) {
    Path classVectorPath = new Path(output, NaiveBayesTrainer.CLASS_VECTORS);
    Path sumVectorPath = new Path(output, NaiveBayesTrainer.SUM_VECTORS);
    Path thetaSumPath = new Path(output, NaiveBayesTrainer.THETA_SUM);

    NaiveBayesModel model = new NaiveBayesModel();
    model.setAlphaI(conf.getFloat(NaiveBayesTrainer.ALPHA_I, 1.0f));

    int featureCount = 0;
    int labelCount = 0;
    // read feature sums and label sums
    for (Pair<Text,VectorWritable> record
         : new SequenceFileIterable<Text, VectorWritable>(sumVectorPath, true, conf)) {
      Text key = record.getFirst();
      VectorWritable value = record.getSecond();
      if (key.toString().equals(BayesConstants.FEATURE_SUM)) {
        model.setFeatureSum(value.get());
        featureCount = value.get().getNumNondefaultElements();
        model.setVocabCount(featureCount);       
      } else  if (key.toString().equals(BayesConstants.LABEL_SUM)) {
        model.setLabelSum(value.get());
        model.setTotalSum(value.get().zSum());
        labelCount = value.get().size();
      }
    }

    // read the class matrix
    Matrix matrix = new SparseMatrix(new int[] {labelCount, featureCount});
    for (Pair<IntWritable,VectorWritable> record
         : new SequenceFileIterable<IntWritable,VectorWritable>(classVectorPath, true, conf)) {
      IntWritable label = record.getFirst();
      VectorWritable value = record.getSecond();
      matrix.assignRow(label.get(), value.get());
    }
    
    model.setWeightMatrix(matrix);

    // read theta normalizer
    for (Pair<Text,VectorWritable> record
         : new SequenceFileIterable<Text,VectorWritable>(thetaSumPath, true, conf)) {
      Text key = record.getFirst();
      VectorWritable value = record.getSecond();
      if (key.toString().equals(BayesConstants.LABEL_THETA_NORMALIZER)) {
        model.setPerlabelThetaNormalizer(value.get());
      }
    }

    return model;
  }
  
  public static void validate(NaiveBayesModel model) {
    if (model == null) {
      return; // empty models are valid
    }

    Preconditions.checkArgument(model.getAlphaI() > 0, "Error: AlphaI has to be greater than 0!");
    Preconditions.checkArgument(model.getVocabCount() > 0, "Error: The vocab count has to be greater than 0!");
    Preconditions.checkArgument(model.getTotalSum() > 0, "Error: The vocab count has to be greater than 0!");
    Preconditions.checkArgument(model.getLabelSum() != null && model.getLabelSum().getNumNondefaultElements() > 0,
        "Error: The number of labels has to be greater than 0 and defined!");
    Preconditions.checkArgument(model.getPerlabelThetaNormalizer() != null &&
        model.getPerlabelThetaNormalizer().getNumNondefaultElements() > 0,
        "Error: The number of theta normalizers has to be greater than 0 or defined!");
    Preconditions.checkArgument(model.getFeatureSum() != null && model.getFeatureSum().getNumNondefaultElements() > 0,
        "Error: The number of features has to be greater than 0 or defined!");
  }
}
