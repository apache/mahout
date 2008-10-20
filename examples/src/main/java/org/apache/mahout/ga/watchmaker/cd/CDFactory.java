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

import org.uncommons.watchmaker.framework.factories.AbstractCandidateFactory;

import java.util.Random;

/**
 * Factory used by Watchmaker to generate the initial population.
 */
public class CDFactory extends AbstractCandidateFactory<CDRule> {

  private final double threshold;

  /**
   * @param threshold condition activation threshold
   */
  public CDFactory(double threshold) {
    this.threshold = threshold;
  }

  @Override
  protected CDRule generateRandomCandidate(Random rng) {
    return new CDRule(threshold, rng);
  }
}
