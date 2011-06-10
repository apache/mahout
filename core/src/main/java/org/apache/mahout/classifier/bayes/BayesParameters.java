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

package org.apache.mahout.classifier.bayes;

import java.io.IOException;

import org.apache.mahout.common.Parameters;

/**
 * BayesParameter used for passing parameters to the Map/Reduce Jobs parameters include gramSize,
 */
public final class BayesParameters extends Parameters {

  private static final String DEFAULT_MIN_SUPPORT = "-1";
  private static final String DEFAULT_MIN_DF = "-1";

  public BayesParameters() {

  }

  public BayesParameters(String serializedString) throws IOException {
    super(parseParams(serializedString));
  }

  public int getGramSize() {
    return Integer.parseInt(get("gramSize"));
  }

  public void setGramSize(int gramSize) {
    set("gramSize", Integer.toString(gramSize));
  }

  public int getMinSupport() {
    return Integer.parseInt(get("minSupport", DEFAULT_MIN_SUPPORT));
  }
  
  public void setMinSupport(int minSupport) {
    set("minSupport", Integer.toString(minSupport));
  }

  public int getMinDF() {
    return Integer.parseInt(get("minDf", DEFAULT_MIN_DF));
  }
  
  public void setMinDF(int minDf) {
    set("minDf", Integer.toString(minDf)); 
  }

  public boolean isSkipCleanup() {
    return Boolean.parseBoolean(get("skipCleanup", "false"));
  }
  
  public void setSkipCleanup(boolean b) {
    set("skipCleanup", Boolean.toString(b));
  }

  public String getBasePath() {
    return get("basePath");
  }

  public void setBasePath(String basePath) {
    set("basePath", basePath);
  }

}
