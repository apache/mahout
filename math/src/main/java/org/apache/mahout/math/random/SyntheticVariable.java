/*
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

package org.apache.mahout.math.random;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;

import java.util.Iterator;
import java.util.Map;

/**
 * Given a summary of a variable, use heuristics to make more of it.
 *
 * The summary has the following tab separated fields:
 *
 * <ul>
 * <li> Variable Name - the name of the variable </li>
 * <li> Variable Type - binary or continuous.  Note that discrete variables are labeled continuous. </li>
 * <li> N - number of observations </li>
 * <li> N Miss - Number of missing observations </li>
 * <li> Miss % - Percentage of records that are missing </li>
 * <li> Mean - The mean of the observations </li>
 * <li> Std Dev - The standard deviation of the observations </li>
 * <li> Minimum - The minimum observed value of the variable </li>
 * <li> Maximum - The maximum observed value of the variable </li>
 * <li> 1st Pctl - The 0.01 quantile </li>
 * <li> ... </li>
 * <li> 95th Pctl </li>
 * <li> 99th Pctl - The 0.99 quantile </li>
 * </ul>
 *
 * For all variables, the generator is wrapped in a missing data wrapper.  Beyond that, the following
 * heuristics are used
 *
 * <ul>
 *   <li> binary variables are assumed binary and the mean taken as the probability </li>
 *   <li> variables marked as continuous with min 0 and max < 50 and all integer quantiles are
 *   assumed to be multinomial. Probabilities are assigned to match the quantiles. </li>
 *   <li> all else are assumed continuous and are modelled using the Empirical generator</li>
 * </ul>
 */
public class SyntheticVariable extends AbstractSamplerFunction {

    private Map<String, String> assign(Iterable<String> split) {
    Iterable<String> labels = ImmutableList.of(
      "name", "type", "n", "missing", "missing%", "mean", "std", "min", "max",
      "q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99");

    Map<String, String> r = Maps.newHashMap();
    Iterator<String> i = labels.iterator();
    for (String value : split) {
      r.put(value, i.next());
    }
    return r;
  }

  @Override
  public Double sample() {
    throw new UnsupportedOperationException("Default operation");
  }
}
