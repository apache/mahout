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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;

import java.io.IOException;
import java.util.Iterator;

/**
 * Calculates the information gain for a {@link org.apache.hadoop.io.SequenceFile}.
 * Computes, how 'useful' are the keys when predicting the values.
 * <ul>
 * <li>-i The input sequence file</li>
 * </ul>
 */
public final class InformationGain extends AbstractJob {

  private static final String ENTROPY_FILE = "entropy";
  private static final String CONDITIONAL_ENTROPY_FILE = "conditional_entropy";

  private Path entropyPath;
  private Path conditionalEntropyPath;
  private double entropy;
  private double conditionalEntropy;
  private double informationGain;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Entropy(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    prepareArguments(args);
    calculateEntropy();
    calculateConditionalEntropy();
    calculateInformationGain();
    return 0;
  }

  public double getEntropy() {
    return entropy;
  }

  public double getConditionalEntropy() {
    return conditionalEntropy;
  }

  public double getInformationGain() {
    return informationGain;
  }

  /**
   * Prepares and sets the arguments.
   */
  private void prepareArguments(String[] args) throws IOException {
    addInputOption();
    parseArguments(args);
    entropyPath = new Path(getTempPath(), ENTROPY_FILE + '-' + System.currentTimeMillis());
    conditionalEntropyPath = new Path(getTempPath(), CONDITIONAL_ENTROPY_FILE + '-' + System.currentTimeMillis());
  }

  private void calculateEntropy() throws Exception {
    String[] args = {
      "-i", getInputPath().toString(),
      "-o", entropyPath.toString(),
      "-s", "value",
      "--tempDir", getTempPath().toString(),
    };
    ToolRunner.run(new Entropy(), args);
    entropy = readDoubleFromPath(entropyPath);
  }

  private void calculateConditionalEntropy() throws Exception {
    String[] args = {
      "-i", getInputPath().toString(),
      "-o", conditionalEntropyPath.toString(),
      "--tempDir", getTempPath().toString(),
    };
    ToolRunner.run(new ConditionalEntropy(), args);
    conditionalEntropy = readDoubleFromPath(conditionalEntropyPath);
  }

  private void calculateInformationGain() {
    informationGain = entropy - conditionalEntropy;
  }

  private static double readDoubleFromPath(Path path) throws IOException {
    Iterator<DoubleWritable> iteratorNodes =
        new SequenceFileDirValueIterator<DoubleWritable>(path,
                                                         PathType.LIST,
                                                         PathFilters.logsCRCFilter(),
                                                         null,
                                                         false,
                                                         new Configuration());
    if (!iteratorNodes.hasNext()) {
      throw new IllegalArgumentException("Can't read double value from " + path.toString());
    }
    return iteratorNodes.next().get();
  }

}
