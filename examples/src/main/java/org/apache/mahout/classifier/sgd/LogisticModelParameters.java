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
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * Encapsulates everything we need to know about a model and how it reads and vectorizes its input.
 * This encapsulation allows us to coherently save and restore a model from a file.  This also
 * allows us to keep command line arguments that affect learning in a coherent way.
 */
public class LogisticModelParameters implements Writable {
  private String targetVariable;
  private Map<String, String> typeMap;
  private int numFeatures;
  private boolean useBias;
  private int maxTargetCategories;
  private List<String> targetCategories;
  private double lambda;
  private double learningRate;
  private CsvRecordFactory csv;
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

  /**
   * Saves a model to an output stream.
   */
  public void saveTo(OutputStream out) throws IOException {
    Closeables.close(lr, false);
    targetCategories = getCsvRecordFactory().getTargetCategories();
    write(new DataOutputStream(out));
  }

  /**
   * Reads a model from a stream.
   */
  public static LogisticModelParameters loadFrom(InputStream in) throws IOException {
    LogisticModelParameters result = new LogisticModelParameters();
    result.readFields(new DataInputStream(in));
    return result;
  }

  /**
   * Reads a model from a file.
   * @throws IOException If there is an error opening or closing the file.
   */
  public static LogisticModelParameters loadFrom(File in) throws IOException {
    InputStream input = new FileInputStream(in);
    try {
      return loadFrom(input);
    } finally {
      Closeables.close(input, true);
    }
  }


  @Override
  public void write(DataOutput out) throws IOException {
    out.writeUTF(targetVariable);
    out.writeInt(typeMap.size());
    for (Map.Entry<String,String> entry : typeMap.entrySet()) {
      out.writeUTF(entry.getKey());
      out.writeUTF(entry.getValue());
    }
    out.writeInt(numFeatures);
    out.writeBoolean(useBias);
    out.writeInt(maxTargetCategories);

    if (targetCategories == null) {
      out.writeInt(0);
    } else {
      out.writeInt(targetCategories.size());
      for (String category : targetCategories) {
        out.writeUTF(category);
      }
    }
    out.writeDouble(lambda);
    out.writeDouble(learningRate);
    // skip csv
    lr.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    targetVariable = in.readUTF();
    int typeMapSize = in.readInt();
    typeMap = Maps.newHashMapWithExpectedSize(typeMapSize);
    for (int i = 0; i < typeMapSize; i++) {
      String key = in.readUTF();
      String value = in.readUTF();
      typeMap.put(key, value);
    }
    numFeatures = in.readInt();
    useBias = in.readBoolean();
    maxTargetCategories = in.readInt();
    int targetCategoriesSize = in.readInt();
    targetCategories = Lists.newArrayListWithCapacity(targetCategoriesSize);
    for (int i = 0; i < targetCategoriesSize; i++) {
      targetCategories.add(in.readUTF());
    }
    lambda = in.readDouble();
    learningRate = in.readDouble();
    csv = null;
    lr = new OnlineLogisticRegression();
    lr.readFields(in);
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

  public List<String> getTargetCategories() {
    return this.targetCategories;
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

  public void setTypeMap(Map<String, String> map) {
    this.typeMap = map;
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
}
