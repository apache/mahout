/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.classifier.mlp;

import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

/**
 * The functions that will be used by NeuralNetwork.
 */
public class NeuralNetworkFunctions {

  /**
   * The derivation of identity function (f(x) = x).
   */
  public static DoubleFunction derivativeIdentityFunction = new DoubleFunction() {
    @Override
    public double apply(double x) {
      return 1;
    }
  };

  /**
   * The derivation of minus squared function (f(t, o) = (o - t)^2).
   */
  public static DoubleDoubleFunction derivativeMinusSquared = new DoubleDoubleFunction() {
    @Override
    public double apply(double target, double output) {
      return 2 * (output - target);
    }
  };

  /**
   * The cross entropy function (f(t, o) = -t * log(o) - (1 - t) * log(1 - o)).
   */
  public static DoubleDoubleFunction crossEntropy = new DoubleDoubleFunction() {
    @Override
    public double apply(double target, double output) {
      return -target * Math.log(output) - (1 - target) * Math.log(1 - output);
    }
  };

  /**
   * The derivation of cross entropy function (f(t, o) = -t * log(o) - (1 - t) *
   * log(1 - o)).
   */
  public static DoubleDoubleFunction derivativeCrossEntropy = new DoubleDoubleFunction() {
    @Override
    public double apply(double target, double output) {
      double adjustedTarget = target;
      double adjustedActual = output;
      if (adjustedActual == 1) {
        adjustedActual = 0.999;
      } else if (output == 0) {
        adjustedActual = 0.001;
      }
      if (adjustedTarget == 1) {
        adjustedTarget = 0.999;
      } else if (adjustedTarget == 0) {
        adjustedTarget = 0.001;
      }
      return -adjustedTarget / adjustedActual + (1 - adjustedTarget) / (1 - adjustedActual);
    }
  };

  /**
   * Get the corresponding function by its name.
   * Currently supports: "Identity", "Sigmoid".
   * 
   * @param function The name of the function.
   * @return The corresponding double function.
   */
  public static DoubleFunction getDoubleFunction(String function) {
    if (function.equalsIgnoreCase("Identity")) {
      return Functions.IDENTITY;
    } else if (function.equalsIgnoreCase("Sigmoid")) {
      return Functions.SIGMOID;
    } else {
      throw new IllegalArgumentException("Function not supported.");
    }
  }

  /**
   * Get the derivation double function by the name.
   * Currently supports: "Identity", "Sigmoid".
   * 
   * @param function The name of the function.
   * @return The double function.
   */
  public static DoubleFunction getDerivativeDoubleFunction(String function) {
    if (function.equalsIgnoreCase("Identity")) {
      return derivativeIdentityFunction;
    } else if (function.equalsIgnoreCase("Sigmoid")) {
      return Functions.SIGMOIDGRADIENT;
    } else {
      throw new IllegalArgumentException("Function not supported.");
    }
  }

  /**
   * Get the corresponding double-double function by the name.
   * Currently supports: "Minus_Squared", "Cross_Entropy".
   * 
   * @param function The name of the function.
   * @return The double-double function.
   */
  public static DoubleDoubleFunction getDoubleDoubleFunction(String function) {
    if (function.equalsIgnoreCase("Minus_Squared")) {
      return Functions.MINUS_SQUARED;
    } else if (function.equalsIgnoreCase("Cross_Entropy")) {
      return derivativeCrossEntropy;
    } else {
      throw new IllegalArgumentException("Function not supported.");
    }
  }

  /**
   * Get the corresponding derivation of double double function by the name.
   * Currently supports: "Minus_Squared", "Cross_Entropy".
   * 
   * @param function The name of the function.
   * @return The double-double-function.
   */
  public static DoubleDoubleFunction getDerivativeDoubleDoubleFunction(String function) {
    if (function.equalsIgnoreCase("Minus_Squared")) {
      return derivativeMinusSquared;
    } else if (function.equalsIgnoreCase("Cross_Entropy")) {
      return derivativeCrossEntropy;
    } else {
      throw new IllegalArgumentException("Function not supported.");
    }
  }

}