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

import org.uncommons.watchmaker.framework.Probability;
import org.uncommons.watchmaker.framework.operators.AbstractCrossover;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Crossover operator.
 */
public class CDCrossover extends AbstractCrossover<CDRule> {

  public CDCrossover(int crossoverPoints) {
    super(crossoverPoints);
  }

  public CDCrossover(int crossoverPoints, Probability crossoverProbability) {
    super(crossoverPoints, crossoverProbability);
  }
  
  @Override
  protected List<CDRule> mate(CDRule parent1, CDRule parent2,
      int numberOfCrossoverPoints, Random rng) {
    if (parent1.getNbConditions() != parent2.getNbConditions())
    {
        throw new IllegalArgumentException("Cannot perform cross-over with parents of different size.");
    }
    CDRule offspring1 = new CDRule(parent1);
    CDRule offspring2 = new CDRule(parent2);
    // Apply as many cross-overs as required.
    for (int i = 0; i < numberOfCrossoverPoints; i++)
    {
        // Cross-over index is always greater than zero and less than
        // the length of the parent so that we always pick a point that
        // will result in a meaningful cross-over.
        int crossoverIndex = (1 + rng.nextInt(parent1.getNbConditions() - 1));
        for (int j = 0; j < crossoverIndex; j++)
        {
          swap(offspring1, offspring2, j);
        }
    }
    
    List<CDRule> result = new ArrayList<CDRule>(2);
    result.add(offspring1);
    result.add(offspring2);
    return result;
  }

  static void swap(CDRule ind1, CDRule ind2, int index) {

    // swap W
    double dtemp = ind1.getW(index);
    ind1.setW(index, ind2.getW(index));
    ind2.setW(index, dtemp);
    
    // swap O
    boolean btemp = ind1.getO(index);
    ind1.setO(index, ind2.getO(index));
    ind2.setO(index, btemp);
    
    // swap V
    dtemp = ind1.getV(index);
    ind1.setV(index, ind2.getV(index));
    ind2.setV(index, dtemp);
  }
}
