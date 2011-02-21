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

import java.io.IOException;
import java.lang.reflect.Type;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.classifier.naivebayes.trainer.NaiveBayesTrainer;
import org.apache.mahout.math.JsonMatrixAdapter;
import org.apache.mahout.math.JsonVectorAdapter;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;

/**
 * NaiveBayesModel holds the weight Matrix, the feature and label sums and the weight normalizer vectors.
 */
public class NaiveBayesModel implements JsonDeserializer<NaiveBayesModel>, JsonSerializer<NaiveBayesModel> {
 
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
  public static NaiveBayesModel fromMRTrainerOutput(Path output, Configuration conf) throws IOException {
    Path classVectorPath = new Path(output, NaiveBayesTrainer.CLASS_VECTORS);
    Path sumVectorPath = new Path(output, NaiveBayesTrainer.SUM_VECTORS);
    Path thetaSumPath = new Path(output, NaiveBayesTrainer.THETA_SUM);

    NaiveBayesModel model = new NaiveBayesModel();
    model.setAlphaI(conf.getFloat(NaiveBayesTrainer.ALPHA_I, 1.0f));
    
    FileSystem fs = sumVectorPath.getFileSystem(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, sumVectorPath, conf);
    Writable key = new Text();
    VectorWritable value = new VectorWritable();

    int featureCount = 0;
    int labelCount = 0;
    // read feature sums and label sums
    while (reader.next(key, value)) {
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
    reader.close();
    
    // read the class matrix
    reader = new SequenceFile.Reader(fs, classVectorPath, conf);
    IntWritable label = new IntWritable();
    Matrix matrix = new SparseMatrix(new int[] {labelCount, featureCount});
    while (reader.next(label, value)) {
      matrix.assignRow(label.get(), value.get());
    }
    reader.close();
    
    model.setWeightMatrix(matrix);
   
    
    
    reader = new SequenceFile.Reader(fs, thetaSumPath, conf);
    // read theta normalizer
    while (reader.next(key, value)) {
      if (key.toString().equals(BayesConstants.LABEL_THETA_NORMALIZER)) {
        model.setPerlabelThetaNormalizer(value.get());
      }
    }
    reader.close();
    
    return model;
  }
  
  /**
   * Encode this NaiveBayesModel as a JSON string
   *
   * @return String containing the JSON of this model
   */
  public String toJson() {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(NaiveBayesModel.class, this);
    Gson gson = builder.create();
    return gson.toJson(this);
  }

  /**
   * Decode this NaiveBayesModel from a JSON string
   *
   * @param json String containing JSON representation of this model
   * @return Initialized model
   */
  public static NaiveBayesModel fromJson(String json) {
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(NaiveBayesModel.class, new NaiveBayesModel());
    Gson gson = builder.create();
    return gson.fromJson(json, NaiveBayesModel.class);
  }
   
  private static final String MODEL = "NaiveBayesModel";

  @Override
  public JsonElement serialize(NaiveBayesModel model,
                               Type type,
                               JsonSerializationContext context) {
    // now register the builders for matrix / vector
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Matrix.class, new JsonMatrixAdapter());
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    // create a model
    JsonObject json = new JsonObject();
    // first, we add the model
    json.add(MODEL, new JsonPrimitive(gson.toJson(model)));
    return json;
  }

  @Override
  public NaiveBayesModel deserialize(JsonElement json,
                                     Type type,
                                     JsonDeserializationContext context) {
    // register the builders for matrix / vector
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(Matrix.class, new JsonMatrixAdapter());
    builder.registerTypeAdapter(Vector.class, new JsonVectorAdapter());
    Gson gson = builder.create();
    // now decode the original model
    JsonObject obj = json.getAsJsonObject();
    String modelString = obj.get(MODEL).getAsString();

    // return the model
    return gson.fromJson(modelString, NaiveBayesModel.class);
  }
  
  public static void validate(NaiveBayesModel model) {
    if (model == null) {
      return; // empty models are valid
    }

    if (model.getAlphaI() <= 0) {
      throw new IllegalArgumentException(
          "Error: AlphaI has to be greater than 0!");
    }

    if (model.getVocabCount() <= 0) {
      throw new IllegalArgumentException(
          "Error: The vocab count has to be greater than 0!");
    }

    if (model.getVocabCount() <= 0) {
      throw new IllegalArgumentException(
          "Error: The vocab count has to be greater than 0!");
    }
    
    if (model.getTotalSum() <= 0) {
      throw new IllegalArgumentException(
          "Error: The vocab count has to be greater than 0!");
    }    

    if (model.getLabelSum() == null || model.getLabelSum().getNumNondefaultElements() <= 0) {
      throw new IllegalArgumentException(
          "Error: The number of labels has to be greater than 0 or defined!");
    }  
    
    if (model.getPerlabelThetaNormalizer() == null
        || model.getPerlabelThetaNormalizer().getNumNondefaultElements() <= 0) {
      throw new IllegalArgumentException(
          "Error: The number of theta normalizers has to be greater than 0 or defined!");
    }
    
    if (model.getFeatureSum() == null || model.getFeatureSum().getNumNondefaultElements() <= 0) {
      throw new IllegalArgumentException(
          "Error: The number of features has to be greater than 0 or defined!");
    }
  }
}
