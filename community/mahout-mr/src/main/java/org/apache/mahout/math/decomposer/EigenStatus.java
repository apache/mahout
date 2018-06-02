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

package org.apache.mahout.math.decomposer;

public class EigenStatus {
  private final double eigenValue;
  private final double cosAngle;
  private volatile Boolean inProgress;

  public EigenStatus(double eigenValue, double cosAngle) {
    this(eigenValue, cosAngle, true);
  }

  public EigenStatus(double eigenValue, double cosAngle, boolean inProgress) {
    this.eigenValue = eigenValue;
    this.cosAngle = cosAngle;
    this.inProgress = inProgress;
  }

  public double getCosAngle() {
    return cosAngle;
  }

  public double getEigenValue() {
    return eigenValue;
  }

  public boolean inProgress() {
    return inProgress;
  }

  void setInProgress(boolean status) {
    inProgress = status;
  }
}
