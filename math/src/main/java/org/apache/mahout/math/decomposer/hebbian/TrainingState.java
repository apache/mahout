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

package org.apache.mahout.math.decomposer.hebbian;

import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;


public class TrainingState {
  TrainingState(Matrix eigens, Matrix projections) {
    setCurrentEigens(eigens);
    setTrainingProjections(projections);
    setTrainingIndex(0);
    setHelperVector(new DenseVector("helper", eigens.numRows()));
    setFirstPass(true);
    setStatusProgress(new ArrayList<EigenStatus>());
    setActivationNumerator(0);
    setActivationDenominatorSquared(0);
    setNumEigensProcessed(0);
  }

  private Matrix currentEigens;
  private int numEigensProcessed;
  private List<Double> currentEigenValues;
  private Matrix trainingProjections;
  private int trainingIndex;
  private Vector helperVector;
  private boolean firstPass;
  private List<EigenStatus> statusProgress;
  private double activationNumerator;
  private double activationDenominatorSquared;

  public Vector mostRecentEigen() {
    return getCurrentEigens().getRow(getNumEigensProcessed() - 1);
  }

  public Vector currentTrainingProjection() {
    if (getTrainingProjections().getRow(getTrainingIndex()) == null)
      getTrainingProjections().assignRow(getTrainingIndex(), new DenseVector(getCurrentEigens().numCols()));
    return getTrainingProjections().getRow(getTrainingIndex());
  }

  public Matrix getCurrentEigens() {
    return currentEigens;
  }

  public void setCurrentEigens(Matrix currentEigens) {
    this.currentEigens = currentEigens;
  }

  public int getNumEigensProcessed() {
    return numEigensProcessed;
  }

  public void setNumEigensProcessed(int numEigensProcessed) {
    this.numEigensProcessed = numEigensProcessed;
  }

  public List<Double> getCurrentEigenValues() {
    return currentEigenValues;
  }

  public void setCurrentEigenValues(List<Double> currentEigenValues) {
    this.currentEigenValues = currentEigenValues;
  }

  public Matrix getTrainingProjections() {
    return trainingProjections;
  }

  public void setTrainingProjections(Matrix trainingProjections) {
    this.trainingProjections = trainingProjections;
  }

  public int getTrainingIndex() {
    return trainingIndex;
  }

  public void setTrainingIndex(int trainingIndex) {
    this.trainingIndex = trainingIndex;
  }

  public Vector getHelperVector() {
    return helperVector;
  }

  public void setHelperVector(Vector helperVector) {
    this.helperVector = helperVector;
  }

  public boolean isFirstPass() {
    return firstPass;
  }

  public void setFirstPass(boolean firstPass) {
    this.firstPass = firstPass;
  }

  public List<EigenStatus> getStatusProgress() {
    return statusProgress;
  }

  public void setStatusProgress(List<EigenStatus> statusProgress) {
    this.statusProgress = statusProgress;
  }

  public double getActivationNumerator() {
    return activationNumerator;
  }

  public void setActivationNumerator(double activationNumerator) {
    this.activationNumerator = activationNumerator;
  }

  public double getActivationDenominatorSquared() {
    return activationDenominatorSquared;
  }

  public void setActivationDenominatorSquared(double activationDenominatorSquared) {
    this.activationDenominatorSquared = activationDenominatorSquared;
  }
}
