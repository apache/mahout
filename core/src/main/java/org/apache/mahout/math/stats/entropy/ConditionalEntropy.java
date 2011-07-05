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

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.math.VarIntWritable;

import java.io.IOException;

/**
 * A Hadoop job to compute the conditional entropy H(Value|Key) for a sequence file.
 * <ul>
 * <li>-i The input sequence file</li>
 * <li>-o The output sequence file</li>
 * </ul>
 */
public final class ConditionalEntropy extends AbstractJob {

  private long numberItems;

  private Path keyValueCountPath;
  private Path specificConditionalEntropyPath;

  private static final String KEY_VALUE_COUNT_FILE = "key_value_count";
  private static final String SPECIFIC_CONDITIONAL_ENTROPY_FILE = "specific_conditional_entropy";
  static final String NUMBER_ITEMS_PARAM = "items.number";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Entropy(), args);
  }

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    prepareArguments(args);
    groupAndCountByKeyAndValue();
    calculateSpecificConditionalEntropy();
    calculateConditionalEntropy();
    return 0;
  }

  /**
   * Prepares and sets the arguments.
   */
  private void prepareArguments(String[] args) throws IOException {
    addInputOption();
    addOutputOption();
    parseArguments(args);
    keyValueCountPath = new Path(getTempPath(), KEY_VALUE_COUNT_FILE + '-' + System.currentTimeMillis());
    specificConditionalEntropyPath =
        new Path(getTempPath(), SPECIFIC_CONDITIONAL_ENTROPY_FILE + '_' + System.currentTimeMillis());
  }

  /**
   * Groups and counts by key and value.
   * SQL-like: SELECT key, value, COUNT(*) FROM x GROUP BY key, value
   */
  private void groupAndCountByKeyAndValue() throws IOException, ClassNotFoundException, InterruptedException {

    Job job = prepareJob(getInputPath(), keyValueCountPath, SequenceFileInputFormat.class,
        GroupAndCountByKeyAndValueMapper.class, StringTuple.class, VarIntWritable.class, VarIntSumReducer.class,
        StringTuple.class, VarIntWritable.class, SequenceFileOutputFormat.class);
    job.setCombinerClass(VarIntSumReducer.class);
    job.waitForCompletion(true);

    numberItems =
        job.getCounters().findCounter("org.apache.hadoop.mapred.Task$Counter", "MAP_INPUT_RECORDS").getValue();

  }

  /**
   * Calculates the specific conditional entropy which is H(Y|X).
   * Needs the number of all items for normalizing.
   */
  private void calculateSpecificConditionalEntropy() throws IOException, ClassNotFoundException, InterruptedException {

    Job job = prepareJob(keyValueCountPath, specificConditionalEntropyPath, SequenceFileInputFormat.class,
        SpecificConditionalEntropyMapper.class, Text.class, VarIntWritable.class,
        SpecificConditionalEntropyReducer.class, Text.class, DoubleWritable.class,
        SequenceFileOutputFormat.class);
    job.getConfiguration().set(NUMBER_ITEMS_PARAM, String.valueOf(numberItems));
    job.waitForCompletion(true);

  }

  /**
   * Sums the calculated specific conditional entropy. Output is in the value.
   */
  private void calculateConditionalEntropy() throws IOException, ClassNotFoundException, InterruptedException {

    Job job = prepareJob(specificConditionalEntropyPath, getOutputPath(), SequenceFileInputFormat.class,
        CalculateSpecificConditionalEntropyMapper.class, NullWritable.class, DoubleWritable.class,
        DoubleSumReducer.class, NullWritable.class, DoubleWritable.class,
        SequenceFileOutputFormat.class);
    job.setCombinerClass(DoubleSumReducer.class);
    job.waitForCompletion(true);

  }

}
