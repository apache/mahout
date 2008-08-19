package org.apache.mahout.classifier;

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

/**
 * Result of a Document Classification. The label and the associated score(Usually probabilty)
 */
public class ClassifierResult {
  private String label;
  private float score;

  public ClassifierResult() {
  }

  public ClassifierResult(String label, float score) {
    this.label = label;
    this.score = score;
  }

  public ClassifierResult(String label) {
    this.label = label;
  }

  public String getLabel() {
    return label;
  }  

  public float getScore() {
    return score;
  }
  
  public void setLabel(String label) {
    this.label = label;
  }  

  public void setScore(float score) {
    this.score = score;
  }

  public String toString() {
    return "ClassifierResult{" +
            "category='" + label + '\'' +
            ", score=" + score +
            '}';
  }
}
