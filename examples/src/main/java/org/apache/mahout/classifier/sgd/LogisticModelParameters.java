/*
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

package org.apache.mahout.classifier.sgd;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.InstanceCreator;
import com.google.gson.JsonArray;
import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.JsonSerializationContext;
import com.google.gson.JsonSerializer;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Encapsulates everything we need to know about a model and how it reads and vectorizes its input.
 * This encapsulation allows us to coherently save and restore a model from a file.  This also
 * allows us to keep command line arguments that affect learning in a coherent way.
 */
public class LogisticModelParameters {
  private String targetVariable;
  private Map<String, String> typeMap;
  private int numFeatures;
  private boolean useBias;
  private int maxTargetCategories;
  private List<String> targetCategories;
  private double lambda;
  private double learningRate;
  private transient CsvRecordFactory csv;
  private OnlineLogisticRegression lr;

  /**
   * Returns a CsvRecordFactory compatible with this logistic model.  The reason that this is tied
   * in here is so that we have access to the list of target categories when it comes time to save
   * the model.  If the input isn't CSV, then calling setTargetCategories before calling saveTo will
   * suffice.
   *
   * @return The CsvRecordFactory.
   */
  public CsvRecordFactory getCsvRecordFactory() {
    if (csv == null) {
      csv = new CsvRecordFactory(getTargetVariable(), getTypeMap())
              .maxTargetValue(getMaxTargetCategories())
              .includeBiasTerm(useBias());
      if (targetCategories != null) {
        csv.defineTargetCategories(targetCategories);
      }
    }
    return csv;
  }

  /**
   * Creates a logistic regression trainer using the parameters collected here.
   *
   * @return The newly allocated OnlineLogisticRegression object
   */
  public OnlineLogisticRegression createRegression() {
    if (lr == null) {
      lr = new OnlineLogisticRegression(getMaxTargetCategories(), getNumFeatures(), new L1())
              .lambda(getLambda())
              .learningRate(getLearningRate())
              .alpha(1 - 1.0e-3);
    }
    return lr;
  }

  public static void saveModel(Writer out,
                               OnlineLogisticRegression model,
                               List<String> targetCategories) throws IOException {
    LogisticModelParameters x = new LogisticModelParameters();
    x.setTargetCategories(targetCategories);
    x.setLambda(model.getLambda());
    x.setLearningRate(model.currentLearningRate());
    x.setNumFeatures(model.numFeatures());
    x.setUseBias(true);
    x.setTargetCategories(targetCategories);
    x.saveTo(out);
  }

  /**
   * Saves a model in JSON format.  This includes the current state of the logistic regression
   * trainer and the dictionary for the target categories.
   *
   * @param out Where to write the model.
   * @throws IOException If we can't write the model.
   */
  public void saveTo(Writer out) throws IOException {
    if (lr != null) {
      lr.close();
    }
    targetCategories = csv.getTargetCategories();
    Gson gson = ModelSerializer.gson();

    String savedForm = gson.toJson(this);
    out.write(savedForm);
  }

  /**
   * Reads a model in JSON format.
   *
   * @param in Where to read the model from.
   * @return The LogisticModelParameters object that we read.
   */
  public static LogisticModelParameters loadFrom(Reader in) {
    GsonBuilder gb = new GsonBuilder();
    gb.registerTypeAdapter(Matrix.class, new MatrixTypeAdapter());
    return gb.create().fromJson(in, LogisticModelParameters.class);
  }

  /**
   * Reads a model in JSON format from a File.
   *
   * @param in Where to read the model from.
   * @return The LogisticModelParameters object that we read.
   * @throws IOException If there is an error opening or closing the file.
   */
  public static LogisticModelParameters loadFrom(File in) throws IOException {
    InputStreamReader input = new FileReader(in);
    try {
      return loadFrom(input);
    } finally {
      input.close();
    }
  }

  /**
   * Sets the types of the predictors.  This will later be used when reading CSV data.  If you don't
   * use the CSV data and convert to vectors on your own, you don't need to call this.
   *
   * @param predictorList The list of variable names.
   * @param typeList      The list of types in the format preferred by CsvRecordFactory.
   */
  public void setTypeMap(Iterable<String> predictorList, List<String> typeList) {
    Preconditions.checkArgument(!typeList.isEmpty(), "Must have at least one type specifier");
    typeMap = Maps.newHashMap();
    Iterator<String> iTypes = typeList.iterator();
    String lastType = null;
    for (Object x : predictorList) {
      // type list can be short .. we just repeat last spec
      if (iTypes.hasNext()) {
        lastType = iTypes.next();
      }
      typeMap.put(x.toString(), lastType);
    }
  }

  /**
   * Sets the target variable.  If you don't use the CSV record factory, then this is irrelevant.
   *
   * @param targetVariable The name of the target variable.
   */
  public void setTargetVariable(String targetVariable) {
    this.targetVariable = targetVariable;
  }

  /**
   * Sets the number of target categories to be considered.
   *
   * @param maxTargetCategories The number of target categories.
   */
  public void setMaxTargetCategories(int maxTargetCategories) {
    this.maxTargetCategories = maxTargetCategories;
  }

  public void setNumFeatures(int numFeatures) {
    this.numFeatures = numFeatures;
  }

  public void setTargetCategories(List<String> targetCategories) {
    this.targetCategories = targetCategories;
    maxTargetCategories = targetCategories.size();
  }

  public void setUseBias(boolean useBias) {
    this.useBias = useBias;
  }

  public boolean useBias() {
    return useBias;
  }

  public String getTargetVariable() {
    return targetVariable;
  }

  public Map<String, String> getTypeMap() {
    return typeMap;
  }

  public int getNumFeatures() {
    return numFeatures;
  }

  public int getMaxTargetCategories() {
    return maxTargetCategories;
  }

  public double getLambda() {
    return lambda;
  }

  public void setLambda(double lambda) {
    this.lambda = lambda;
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  /**
   * Tells GSON how to (de)serialize a Mahout matrix.  We assume on deserialization that
   * the matrix is dense.
   */
  public static class MatrixTypeAdapter
    implements JsonDeserializer<Matrix>, JsonSerializer<Matrix>, InstanceCreator<Matrix> {
    @Override
    public JsonElement serialize(Matrix m, Type type, JsonSerializationContext jsonSerializationContext) {
      JsonObject r = new JsonObject();
      r.add("rows", new JsonPrimitive(m.numRows()));
      r.add("cols", new JsonPrimitive(m.numCols()));
      JsonArray v = new JsonArray();
      for (int row = 0; row < m.numRows(); row++) {
        JsonArray rowData = new JsonArray();
        for (int col = 0; col < m.numCols(); col++) {
          rowData.add(new JsonPrimitive(m.get(row, col)));
        }
        v.add(rowData);
      }
      r.add("data", v);
      return r;
    }

    @Override
    public Matrix deserialize(JsonElement x, Type type, JsonDeserializationContext jsonDeserializationContext) {
      JsonObject data = x.getAsJsonObject();
      Matrix r = new DenseMatrix(data.get("rows").getAsInt(), data.get("cols").getAsInt());
      int i = 0;
      for (JsonElement row : data.get("data").getAsJsonArray()) {
        int j = 0;
        for (JsonElement element : row.getAsJsonArray()) {
          r.set(i, j, element.getAsDouble());
          j++;
        }
        i++;
      }
      return r;
    }

    @Override
    public Matrix createInstance(Type type) {
      return new DenseMatrix();
    }
  }
}
