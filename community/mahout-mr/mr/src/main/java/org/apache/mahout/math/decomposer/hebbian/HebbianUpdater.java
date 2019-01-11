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


import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.PlusMult;

public class HebbianUpdater implements EigenUpdater {

  @Override
  public void update(Vector pseudoEigen,
                     Vector trainingVector,
                     TrainingState currentState) {
    double trainingVectorNorm = trainingVector.norm(2);
    int numPreviousEigens = currentState.getNumEigensProcessed();
    if (numPreviousEigens > 0 && currentState.isFirstPass()) {
      updateTrainingProjectionsVector(currentState, trainingVector, numPreviousEigens - 1);
    }
    if (currentState.getActivationDenominatorSquared() == 0 || trainingVectorNorm == 0) {
      if (currentState.getActivationDenominatorSquared() == 0) {
        pseudoEigen.assign(trainingVector, new PlusMult(1));
        currentState.setHelperVector(currentState.currentTrainingProjection().clone());
        double helperNorm = currentState.getHelperVector().norm(2);
        currentState.setActivationDenominatorSquared(trainingVectorNorm * trainingVectorNorm - helperNorm * helperNorm);
      }
      return;
    }
    currentState.setActivationNumerator(pseudoEigen.dot(trainingVector));
    currentState.setActivationNumerator(
        currentState.getActivationNumerator()
            - currentState.getHelperVector().dot(currentState.currentTrainingProjection()));

    double activation = currentState.getActivationNumerator()
        / Math.sqrt(currentState.getActivationDenominatorSquared());
    currentState.setActivationDenominatorSquared(
        currentState.getActivationDenominatorSquared()
            + 2 * activation * currentState.getActivationNumerator()
            + activation * activation
                * (trainingVector.getLengthSquared() - currentState.currentTrainingProjection().getLengthSquared()));
    if (numPreviousEigens > 0) {
      currentState.getHelperVector().assign(currentState.currentTrainingProjection(), new PlusMult(activation));
    }
    pseudoEigen.assign(trainingVector, new PlusMult(activation));
  }

  private static void updateTrainingProjectionsVector(TrainingState state,
                                                      Vector trainingVector,
                                                      int previousEigenIndex) {
    Vector previousEigen = state.mostRecentEigen();
    Vector currentTrainingVectorProjection = state.currentTrainingProjection();
    double projection = previousEigen.dot(trainingVector);
    currentTrainingVectorProjection.set(previousEigenIndex, projection);
  }

}
