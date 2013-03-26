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

package org.apache.mahout.math.hadoop.stats;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;

import java.io.IOException;

/**
 * Methods for calculating basic stats (mean, variance, stdDev, etc.) in map/reduce
 */
public final class BasicStats {

  private BasicStats() {
  }

  /**
   * Calculate the variance of values stored as
   *
   * @param input    The input file containing the key and the count
   * @param output   The output to store the intermediate values
   * @param baseConf
   * @return The variance (based on sample estimation)
   */
  public static double variance(Path input, Path output,
                                Configuration baseConf)
    throws IOException, InterruptedException, ClassNotFoundException {
    VarianceTotals varianceTotals = computeVarianceTotals(input, output, baseConf);
    return varianceTotals.computeVariance();
  }

  /**
   * Calculate the variance by a predefined mean of values stored as
   *
   * @param input    The input file containing the key and the count
   * @param output   The output to store the intermediate values
   * @param mean The mean based on which to compute the variance
   * @param baseConf
   * @return The variance (based on sample estimation)
   */
  public static double varianceForGivenMean(Path input, Path output, double mean,
                                Configuration baseConf)
    throws IOException, InterruptedException, ClassNotFoundException {
    VarianceTotals varianceTotals = computeVarianceTotals(input, output, baseConf);
    return varianceTotals.computeVarianceForGivenMean(mean);
  }

  private static VarianceTotals computeVarianceTotals(Path input, Path output,
                                Configuration baseConf) throws IOException, InterruptedException,
          ClassNotFoundException {
    Configuration conf = new Configuration(baseConf);
    conf.set("io.serializations",
                    "org.apache.hadoop.io.serializer.JavaSerialization,"
                            + "org.apache.hadoop.io.serializer.WritableSerialization");
    Job job = HadoopUtil.prepareJob(input, output, SequenceFileInputFormat.class,
        StandardDeviationCalculatorMapper.class, IntWritable.class, DoubleWritable.class,
        StandardDeviationCalculatorReducer.class, IntWritable.class, DoubleWritable.class,
        SequenceFileOutputFormat.class, conf);
    HadoopUtil.delete(conf, output);
    job.setCombinerClass(StandardDeviationCalculatorReducer.class);
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }

    // Now extract the computed sum
    Path filesPattern = new Path(output, "part-*");
    double sumOfSquares = 0;
    double sum = 0;
    double totalCount = 0;
    for (Pair<Writable, Writable> record : new SequenceFileDirIterable<Writable, Writable>(
            filesPattern, PathType.GLOB, null, null, true, conf)) {

      int key = ((IntWritable) record.getFirst()).get();
      if (key == StandardDeviationCalculatorMapper.SUM_OF_SQUARES.get()) {
        sumOfSquares += ((DoubleWritable) record.getSecond()).get();
      } else if (key == StandardDeviationCalculatorMapper.TOTAL_COUNT
              .get()) {
        totalCount += ((DoubleWritable) record.getSecond()).get();
      } else if (key == StandardDeviationCalculatorMapper.SUM
              .get()) {
        sum += ((DoubleWritable) record.getSecond()).get();
      }
    }

    VarianceTotals varianceTotals = new VarianceTotals();
    varianceTotals.setSum(sum);
    varianceTotals.setSumOfSquares(sumOfSquares);
    varianceTotals.setTotalCount(totalCount);

    return varianceTotals;
  }

  /**
   * Calculate the standard deviation
   *
   * @param input    The input file containing the key and the count
   * @param output   The output file to write the counting results to
   * @param baseConf The base configuration
   * @return The standard deviation
   */
  public static double stdDev(Path input, Path output,
                              Configuration baseConf) throws IOException, InterruptedException,
          ClassNotFoundException {
    return Math.sqrt(variance(input, output, baseConf));
  }

  /**
   * Calculate the standard deviation given a predefined mean
   *
   * @param input    The input file containing the key and the count
   * @param output   The output file to write the counting results to
   * @param mean The mean based on which to compute the standard deviation
   * @param baseConf The base configuration
   * @return The standard deviation
   */
  public static double stdDevForGivenMean(Path input, Path output, double mean,
                              Configuration baseConf) throws IOException, InterruptedException,
          ClassNotFoundException {
    return Math.sqrt(varianceForGivenMean(input, output, mean, baseConf));
  }
}
