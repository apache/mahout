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

package org.apache.mahout.ga.watchmaker.cd;

import org.uncommons.watchmaker.framework.EvolutionaryOperator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Mutation operator.
 */
public class CDMutation implements EvolutionaryOperator<CDRule> {

  /** probability of mutating a variable */
  private final double rate;

  /** max size of the change (step-size) for each mutated variable */
  private final double range;

  /**
   * mutation precision. Defines indirectly the minimal step-size and the
   * distribution of mutation steps inside the mutation range.
   */
  private final int k;

  /**
   * 
   * @param rate probability of mutating a variable
   * @param range max step-size for each variable
   * @param k mutation precision
   * 
   * See http://www.geatbx.com/docu/algindex-04.html#P659_42386 real valued mutation
   * for more information about the parameters
   */
  public CDMutation(double rate, double range, int k) {
    if (rate <= 0 || rate > 1)
      throw new IllegalArgumentException("mutation rate must be in ]0, 1]");
    if (range <= 0 || range > 1)
      throw new IllegalArgumentException("mutation range must be in ]0, 1]");
    if (k < 0)
      throw new IllegalArgumentException("mutation precision must be >= 0");

    this.rate = rate;
    this.range = range;
    this.k = k;
  }

  @Override
  @SuppressWarnings("unchecked")
  public <S extends CDRule> List<S> apply(List<S> selectedCandidates, Random rng) {
    List<S> mutatedPopulation = new ArrayList<S>(selectedCandidates.size());
    for (CDRule ind : selectedCandidates) {
      mutatedPopulation.add((S) mutate(ind, rng));
    }
    return mutatedPopulation;
  }

  protected CDRule mutate(CDRule rule, Random rng) {
    DataSet dataset = DataSet.getDataSet();

    for (int condInd = 0; condInd < rule.getNbConditions(); condInd++) {
      if (rng.nextDouble() > rate)
        continue;

      int attrInd = CDRule.attributeIndex(condInd);

      rule.setW(condInd, rndDouble(rule.getW(condInd), 0.0, 1.0, rng));

      if (dataset.isNumerical(attrInd)) {
        rule.setV(condInd, rndDouble(rule.getV(condInd), dataset
            .getMin(attrInd), dataset.getMax(attrInd), rng));
      } else {
        rule.setV(condInd, rndInt(rule.getV(condInd), dataset
            .getNbValues(attrInd), rng));
      }
    }

    return rule;
  }

  /**
   * returns a random double in the interval [min, max ].
   */
  double rndDouble(double value, double min, double max, Random rng) {
    double s = rng.nextDouble() * 2.0 - 1.0; // [-1, +1]
    double r = range * ((max - min) / 2);
    double a = Math.pow(2, -k * rng.nextDouble());
    double stp = s * r * a;

    value += stp;

    // clamp value to [min, max]
    value = Math.max(min, value);
    value = Math.min(max, value);

    return value;
  }

  static int rndInt(double value, int nbcategories, Random rng) {
    return rng.nextInt(nbcategories);
  }
}
