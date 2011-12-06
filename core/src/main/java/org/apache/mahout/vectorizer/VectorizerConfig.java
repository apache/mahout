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

package org.apache.mahout.vectorizer;

import org.apache.hadoop.conf.Configuration;

/**
 * The config for a Vectorizer.  Not all implementations need use all variables.
 */
public final class VectorizerConfig {

  private Configuration conf;
  private String analyzerClassName;
  private String encoderName;
  private boolean sequentialAccess;
  private boolean namedVectors;
  private int cardinality;
  private String encoderClass;
  private String tfDirName;
  private int minSupport;
  private int maxNGramSize;
  private float minLLRValue;
  private float normPower;
  private boolean logNormalize;
  private int numReducers;
  private int chunkSizeInMegabytes;

  public VectorizerConfig(Configuration conf,
                          String analyzerClassName,
                          String encoderClass,
                          String encoderName,
                          boolean sequentialAccess,
                          boolean namedVectors,
                          int cardinality) {
    this.conf = conf;
    this.analyzerClassName = analyzerClassName;
    this.encoderClass = encoderClass;
    this.encoderName = encoderName;
    this.sequentialAccess = sequentialAccess;
    this.namedVectors = namedVectors;
    this.cardinality = cardinality;
  }

  public Configuration getConf() {
    return conf;
  }

  public void setConf(Configuration conf) {
    this.conf = conf;
  }

  public String getAnalyzerClassName() {
    return analyzerClassName;
  }

  public void setAnalyzerClassName(String analyzerClassName) {
    this.analyzerClassName = analyzerClassName;
  }

  public String getEncoderName() {
    return encoderName;
  }

  public void setEncoderName(String encoderName) {
    this.encoderName = encoderName;
  }

  public boolean isSequentialAccess() {
    return sequentialAccess;
  }

  public void setSequentialAccess(boolean sequentialAccess) {
    this.sequentialAccess = sequentialAccess;
  }


  public String getTfDirName() {
    return tfDirName;
  }

  public void setTfDirName(String tfDirName) {
    this.tfDirName = tfDirName;
  }

  public boolean isNamedVectors() {
    return namedVectors;
  }

  public void setNamedVectors(boolean namedVectors) {
    this.namedVectors = namedVectors;
  }

  public int getCardinality() {
    return cardinality;
  }

  public void setCardinality(int cardinality) {
    this.cardinality = cardinality;
  }

  public String getEncoderClass() {
    return encoderClass;
  }

  public void setEncoderClass(String encoderClass) {
    this.encoderClass = encoderClass;
  }

  public int getMinSupport() {
    return minSupport;
  }

  public void setMinSupport(int minSupport) {
    this.minSupport = minSupport;
  }

  public int getMaxNGramSize() {
    return maxNGramSize;
  }

  public void setMaxNGramSize(int maxNGramSize) {
    this.maxNGramSize = maxNGramSize;
  }

  public float getMinLLRValue() {
    return minLLRValue;
  }

  public void setMinLLRValue(float minLLRValue) {
    this.minLLRValue = minLLRValue;
  }

  public float getNormPower() {
    return normPower;
  }

  public void setNormPower(float normPower) {
    this.normPower = normPower;
  }

  public boolean isLogNormalize() {
    return logNormalize;
  }

  public void setLogNormalize(boolean logNormalize) {
    this.logNormalize = logNormalize;
  }

  public int getNumReducers() {
    return numReducers;
  }

  public void setNumReducers(int numReducers) {
    this.numReducers = numReducers;
  }

  public int getChunkSizeInMegabytes() {
    return chunkSizeInMegabytes;
  }

  public void setChunkSizeInMegabytes(int chunkSizeInMegabytes) {
    this.chunkSizeInMegabytes = chunkSizeInMegabytes;
  }
}
