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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.VarIntWritable;

import java.io.IOException;
import java.util.Map;

/**
 * A Hadoop job to compute the entropy of keys or values in a {@link SequenceFile}. Format has to be {@link Text} for
 * key or value.
 * <p/>
 * <ul>
 * <li>-i The input sequence file</li>
 * <li>-o The output sequence file</li>
 * <li>-s The source. Can be \<key\> or \<value\>. Default is \<key\></li>
 * </ul>
 */
public final class Entropy extends AbstractJob {

  private Path tempPath;
  private long numberItems;
  private String source;

  private static final String TEMP_FILE = "temp";
  static final String NUMBER_ITEMS_PARAM = "number.items";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Entropy(), args);
  }

  /**
   * Returns the number of elements in the file. Only works after run.
   *
   * @return The number of processed items
   */
  public long getNumberItems() {
    return numberItems;
  }

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

    prepareArguments(args);
    groupAndCount();
    calculateEntropy();

    return 1;
  }

  /**
   * Prepares and sets the arguments.
   *
   * @param args
   * @throws IOException
   */
  private void prepareArguments(String[] args) throws IOException {

    addInputOption();
    addOutputOption();
    addOption("source", "s", "Sets, if the entropy is calculated for the keys or the values. Can be <key> or <value>"
        , "key");

    Map<String, String> arguments = parseArguments(args);
    source = arguments.get("--source");
    tempPath = new Path(getTempPath(), TEMP_FILE + '-' + System.currentTimeMillis());

  }


  /**
   * Groups the items and counts the occur for each of them.
   * SQL-like: SELECT item, COUNT(*) FROM x GROUP BY item
   *
   * @throws IOException
   * @throws ClassNotFoundException
   * @throws InterruptedException
   */
  private void groupAndCount() throws IOException, ClassNotFoundException, InterruptedException {

    Class<? extends Mapper> mapper = "key".equals(source) ? KeyCounterMapper.class : ValueCounterMapper.class;

    Job job = prepareJob(getInputPath(), tempPath, SequenceFileInputFormat.class, mapper, Text.class,
        VarIntWritable.class, VarIntSumReducer.class, Text.class, VarIntWritable.class,
        SequenceFileOutputFormat.class);
    job.setCombinerClass(VarIntSumReducer.class);
    job.waitForCompletion(true);

    numberItems =
        job.getCounters().findCounter("org.apache.hadoop.mapred.Task$Counter", "MAP_INPUT_RECORDS").getValue();

  }

  /**
   * Calculates the entropy with
   * <p/>
   * H(X) = -sum_i(x_i/n * log_2(x_i/n))  WITH n = sum_i(x_i)
   * = -sum_i(x_i/n * (log_2(x_i) - log_2(n)))
   * = -sum_i(x_i/n * log_2(x_i)) + sum_i(x_i/n * log_2(n))
   * = (n * log_2(n) - sum_i(x_i * log_2(x_i)) / n
   * = log_2(n) - sum_i(x_i * log_2(x_i)) / n
   * = (log(n) - sum_i(x_i * log(x_i)) / n) / log(2)
   */
  private void calculateEntropy() throws IOException, ClassNotFoundException, InterruptedException {

    Job job = prepareJob(tempPath, getOutputPath(), SequenceFileInputFormat.class, CalculateEntropyMapper.class,
        NullWritable.class, DoubleWritable.class, CalculateEntropyReducer.class, NullWritable.class,
        DoubleWritable.class, SequenceFileOutputFormat.class);
    job.getConfiguration().set(NUMBER_ITEMS_PARAM, String.valueOf(numberItems));
    job.setCombinerClass(DoubleSumReducer.class);
    job.waitForCompletion(true);

  }

}
