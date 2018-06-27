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

package org.apache.mahout.classifier.sgd;

final class SGDInfo {

  private double averageLL;
  private double averageCorrect;
  private double step;
  private int[] bumps = {1, 2, 5};

  double getAverageLL() {
    return averageLL;
  }

  void setAverageLL(double averageLL) {
    this.averageLL = averageLL;
  }

  double getAverageCorrect() {
    return averageCorrect;
  }

  void setAverageCorrect(double averageCorrect) {
    this.averageCorrect = averageCorrect;
  }

  double getStep() {
    return step;
  }

  void setStep(double step) {
    this.step = step;
  }

  int[] getBumps() {
    return bumps;
  }

  void setBumps(int[] bumps) {
    this.bumps = bumps;
  }

}
