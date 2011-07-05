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

package org.apache.mahout.math.stats.entropy;

import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;

/**
 * A job to calculate the normalized information gain.
 * <ul>
 * <li>-i The input sequence file</li>
 * </ul>
 */
public final class InformationGainRatio extends AbstractJob {

  private double entropy;
  private double informationGain;
  private double informationGainRatio;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new InformationGainRatio(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    InformationGain job = new InformationGain();
    ToolRunner.run(job, args);
    informationGain = job.getInformationGain();
    entropy = job.getEntropy();
    informationGainRatio = informationGain / entropy;
    return 0;
  }

  public double getEntropy() {
    return entropy;
  }

  public double getInformationGain() {
    return informationGain;
  }

  public double getInformationGainRatio() {
    return informationGainRatio;
  }

}
