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

package org.apache.mahout.ga.watchmaker;

import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.mahout.utils.StringUtils;
import org.uncommons.watchmaker.framework.FitnessEvaluator;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;

/**
 * Generic Mahout distributed evaluator. takes an evaluator and a population and launches a Hadoop job. The job
 * evaluates the fitness of each individual of the population using the given evaluator. Takes care of storing the
 * population into an input file, and loading the fitness from job outputs.
 */
public class MahoutEvaluator {
  private MahoutEvaluator() {
  }

  /**
   * Uses Mahout to evaluate every candidate from the input population using the given evaluator.
   *
   * @param evaluator   FitnessEvaluator to use
   * @param population  input population
   * @param evaluations <code>List&lt;Double&gt;</code> that contains the evaluated fitness for each candidate from the
   *                    input population, sorted in the same order as the candidates.
   */
  public static void evaluate(FitnessEvaluator<?> evaluator, List<?> population,
                              List<Double> evaluations) throws IOException {
    JobConf conf = new JobConf(MahoutEvaluator.class);
    FileSystem fs = FileSystem.get(conf);
    Path inpath = prepareInput(fs, population);
    Path outpath = OutputUtils.prepareOutput(fs);


    configureJob(conf, evaluator, inpath, outpath);
    JobClient.runJob(conf);

    OutputUtils.importEvaluations(fs, conf, outpath, evaluations);
  }

  /**
   * Create the input directory and stores the population in it.
   *
   * @param fs         <code>FileSystem</code> to use
   * @param population population to store
   * @return input <code>Path</code>
   */
  private static Path prepareInput(FileSystem fs, List<?> population)
      throws IOException {
    Path inpath = new Path(fs.getWorkingDirectory(), "input");

    // Delete the input if it already exists
    if (fs.exists(inpath)) {
      fs.delete(inpath, true);
    }

    fs.mkdirs(inpath);

    storePopulation(fs, new Path(inpath, "population"), population);

    return inpath;
  }

  /**
   * Configure the job
   *
   * @param evaluator FitnessEvaluator passed to the mapper
   * @param inpath    input <code>Path</code>
   * @param outpath   output <code>Path</code>
   */
  private static void configureJob(JobConf conf, FitnessEvaluator<?> evaluator,
                                   Path inpath, Path outpath) {
    FileInputFormat.setInputPaths(conf, inpath);
    FileOutputFormat.setOutputPath(conf, outpath);

    conf.setOutputKeyClass(LongWritable.class);
    conf.setOutputValueClass(DoubleWritable.class);

    conf.setMapperClass(EvalMapper.class);
    // no combiner
    // identity reducer
    // TODO do we really need a reducer at all ?

    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    // store the stringified evaluator
    conf.set(EvalMapper.MAHOUT_GA_EVALUATOR, StringUtils.toString(evaluator));
  }

  /**
   * Stores a population of candidates in the output file path.
   *
   * @param fs         FileSystem used to create the output file
   * @param f          output file path
   * @param population population to store
   */
  static void storePopulation(FileSystem fs, Path f, List<?> population)
      throws IOException {
    FSDataOutputStream out = fs.create(f);
    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out));

    try {
      for (Object candidate : population) {
        writer.write(StringUtils.toString(candidate));
        writer.newLine();
      }
    } finally {
      writer.close();
    }
  }

}
