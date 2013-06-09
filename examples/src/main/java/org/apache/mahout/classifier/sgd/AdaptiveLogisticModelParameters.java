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

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.mahout.math.stats.GlobalOnlineAuc;
import org.apache.mahout.math.stats.GroupedOnlineAuc;
import org.apache.mahout.math.stats.OnlineAuc;

public class AdaptiveLogisticModelParameters extends LogisticModelParameters {

  private AdaptiveLogisticRegression alr;
  private int interval = 800;
  private int averageWindow = 500;
  private int threads = 4;
  private String prior = "L1";
  private double priorOption = Double.NaN;
  private String auc = null;

  public AdaptiveLogisticRegression createAdaptiveLogisticRegression() {

    if (alr == null) {
      alr = new AdaptiveLogisticRegression(getMaxTargetCategories(),
                                           getNumFeatures(), createPrior(prior, priorOption));
      alr.setInterval(interval);
      alr.setAveragingWindow(averageWindow);
      alr.setThreadCount(threads);
      alr.setAucEvaluator(createAUC(auc));
    }
    return alr;
  }

  public void checkParameters() {
    if (prior != null) {
      String priorUppercase = prior.toUpperCase(Locale.ENGLISH).trim();
      if (("TP".equals(priorUppercase) || "EBP".equals(priorUppercase)) && Double.isNaN(priorOption)) {
        throw new IllegalArgumentException("You must specify a double value for TPrior and ElasticBandPrior.");
      }
    }
  }

  private static PriorFunction createPrior(String cmd, double priorOption) {
    if (cmd == null) {
      return null;
    }
    if ("L1".equals(cmd.toUpperCase(Locale.ENGLISH).trim())) {
      return new L1();
    }
    if ("L2".equals(cmd.toUpperCase(Locale.ENGLISH).trim())) {
      return new L2();
    }
    if ("UP".equals(cmd.toUpperCase(Locale.ENGLISH).trim())) {
      return new UniformPrior();
    }
    if ("TP".equals(cmd.toUpperCase(Locale.ENGLISH).trim())) {
      return new TPrior(priorOption);
    }
    if ("EBP".equals(cmd.toUpperCase(Locale.ENGLISH).trim())) {
      return new ElasticBandPrior(priorOption);
    }

    return null;
  }

  private static OnlineAuc createAUC(String cmd) {
    if (cmd == null) {
      return null;
    }
    if ("GLOBAL".equals(cmd.toUpperCase(Locale.ENGLISH).trim())) {
      return new GlobalOnlineAuc();
    }
    if ("GROUPED".equals(cmd.toUpperCase(Locale.ENGLISH).trim())) {
      return new GroupedOnlineAuc();
    }
    return null;
  }

  @Override
  public void saveTo(OutputStream out) throws IOException {
    if (alr != null) {
      alr.close();
    }
    setTargetCategories(getCsvRecordFactory().getTargetCategories());
    write(new DataOutputStream(out));
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeUTF(getTargetVariable());
    out.writeInt(getTypeMap().size());
    for (Map.Entry<String, String> entry : getTypeMap().entrySet()) {
      out.writeUTF(entry.getKey());
      out.writeUTF(entry.getValue());
    }
    out.writeInt(getNumFeatures());
    out.writeInt(getMaxTargetCategories());
    out.writeInt(getTargetCategories().size());
    for (String category : getTargetCategories()) {
      out.writeUTF(category);
    }

    out.writeInt(interval);
    out.writeInt(averageWindow);
    out.writeInt(threads);
    out.writeUTF(prior);
    out.writeDouble(priorOption);
    out.writeUTF(auc);

    // skip csv
    alr.write(out);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    setTargetVariable(in.readUTF());
    int typeMapSize = in.readInt();
    Map<String, String> typeMap = new HashMap<String, String>(typeMapSize);
    for (int i = 0; i < typeMapSize; i++) {
      String key = in.readUTF();
      String value = in.readUTF();
      typeMap.put(key, value);
    }
    setTypeMap(typeMap);

    setNumFeatures(in.readInt());
    setMaxTargetCategories(in.readInt());
    int targetCategoriesSize = in.readInt();
    List<String> targetCategories = Lists.newArrayListWithCapacity(targetCategoriesSize);
    for (int i = 0; i < targetCategoriesSize; i++) {
      targetCategories.add(in.readUTF());
    }
    setTargetCategories(targetCategories);

    interval = in.readInt();
    averageWindow = in.readInt();
    threads = in.readInt();
    prior = in.readUTF();
    priorOption = in.readDouble();
    auc = in.readUTF();

    alr = new AdaptiveLogisticRegression();
    alr.readFields(in);
  }


  private static AdaptiveLogisticModelParameters loadFromStream(InputStream in) throws IOException {
    AdaptiveLogisticModelParameters result = new AdaptiveLogisticModelParameters();
    result.readFields(new DataInputStream(in));
    return result;
  }

  public static AdaptiveLogisticModelParameters loadFromFile(File in) throws IOException {
    InputStream input = new FileInputStream(in);
    try {
      return loadFromStream(input);
    } finally {
      Closeables.close(input, true);
    }
  }

  public int getInterval() {
    return interval;
  }

  public void setInterval(int interval) {
    this.interval = interval;
  }

  public int getAverageWindow() {
    return averageWindow;
  }

  public void setAverageWindow(int averageWindow) {
    this.averageWindow = averageWindow;
  }

  public int getThreads() {
    return threads;
  }

  public void setThreads(int threads) {
    this.threads = threads;
  }

  public String getPrior() {
    return prior;
  }

  public void setPrior(String prior) {
    this.prior = prior;
  }

  public String getAuc() {
    return auc;
  }

  public void setAuc(String auc) {
    this.auc = auc;
  }

  public double getPriorOption() {
    return priorOption;
  }

  public void setPriorOption(double priorOption) {
    this.priorOption = priorOption;
  }


}
