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

package org.apache.mahout.math.stats;

/**
 * Utility methods for working with log-likelihood
 */
public final class LogLikelihood {

  private LogLikelihood() {
  }

  /**
   * Calculate the unnormalized Shannon entropy.  This is
   *
   * -sum x_i log x_i / N = -N sum x_i/N log x_i/N
   *
   * where N = sum x_i
   *
   * If the x's sum to 1, then this is the same as the normal
   * expression.  Leaving this un-normalized makes working with
   * counts and computing the LLR easier.
   *
   * @return The entropy value for the elements
   */
  public static double entropy(int... elements) {
    double sum = 0.0;
    double result = 0.0;
    for (int element : elements) {
      if (element < 0) {
        throw new IllegalArgumentException("Should not have negative count for entropy computation: (" + element + ')');
      }
      if (element > 0) {
        result += element * Math.log(element);
        sum += element;
      }
    }
    result -= sum * Math.log(sum);
    return -result;
  }

  /**
   * Calculate the Raw Log-likelihood ratio for two events, call them A and B.  Then we have:
   * <p/>
   * <table border="1" cellpadding="5" cellspacing="0">
   * <tbody><tr><td>&nbsp;</td><td>Event A</td><td>Everything but A</td></tr>
   * <tr><td>Event B</td><td>A and B together (k_11)</td><td>B, but not A (k_12)</td></tr>
   * <tr><td>Everything but B</td><td>A without B (k_21)</td><td>Neither A nor B (k_22)</td></tr></tbody>
   * </table>
   *
   * @param k11 The number of times the two events occurred together
   * @param k12 The number of times the second event occurred WITHOUT the first event
   * @param k21 The number of times the first event occurred WITHOUT the second event
   * @param k22 The number of times something else occurred (i.e. was neither of these events
   * @return The raw log-likelihood ratio
   *
   * <p/>
   * Credit to http://tdunning.blogspot.com/2008/03/surprise-and-coincidence.html for the table and the descriptions.
   */
  public static double logLikelihoodRatio(int k11, int k12, int k21, int k22) {
    // note that we have counts here, not probabilities, and that the entropy is not normalized.
    double rowEntropy = entropy(k11, k12) + entropy(k21, k22);
    double columnEntropy = entropy(k11, k21) + entropy(k12, k22);
    double matrixEntropy = entropy(k11, k12, k21, k22);
    if (rowEntropy + columnEntropy > matrixEntropy) {
      // round off error
      return 0.0;
    }
    return 2.0 * (matrixEntropy - rowEntropy - columnEntropy);
  }
  
  /** 
   * Calculate the root log-likelihood ratio for two events.
   * See {@link #logLikelihoodRatio(int, int, int, int)}.

   * @param k11 The number of times the two events occurred together
   * @param k12 The number of times the second event occurred WITHOUT the first event
   * @param k21 The number of times the first event occurred WITHOUT the second event
   * @param k22 The number of times something else occurred (i.e. was neither of these events
   * @return The root log-likelihood ratio
   * 
   * <p/>
   * See discussion of raw vs. root LLR at 
   * http://www.lucidimagination.com/search/document/6dc8709e65a7ced1/llr_scoring_question
   */
  public static double rootLogLikelihoodRatio(int k11, int k12, int k21, int k22) {
    double llr = logLikelihoodRatio(k11, k12, k21, k22);
    double sqrt = Math.sqrt(llr);
    if (((double) k11 / (k11 + k12)) < ((double) k21 / (k21 + k22))) {
      sqrt = -sqrt;
    }
    return sqrt;
  }
}
