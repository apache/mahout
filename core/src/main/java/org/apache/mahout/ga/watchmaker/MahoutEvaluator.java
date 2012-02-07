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

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Collection;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.StringUtils;
import org.uncommons.watchmaker.framework.FitnessEvaluator;

/**
 * Generic Mahout distributed evaluator. takes an evaluator and a population and launches a Hadoop job. The
 * job evaluates the fitness of each individual of the population using the given evaluator. Takes care of
 * storing the population into an input file, and loading the fitness from job outputs.
 */
public final class MahoutEvaluator {

  private MahoutEvaluator() {
  }
  
  /**
   * Uses Mahout to evaluate every candidate from the input population using the given evaluator.
   *
   * @param evaluator
   *          FitnessEvaluator to use
   * @param population
   *          input population
   * @param evaluations
   *          {@code List<Double>} that contains the evaluated fitness for each candidate from the
   *          input population, sorted in the same order as the candidates.
   */
  public static void evaluate(FitnessEvaluator<?> evaluator,
                              Iterable<?> population,
                              Collection<Double> evaluations,
                              Path input,
                              Path output) throws IOException, ClassNotFoundException, InterruptedException {

    Job job = new Job();
    job.setJarByClass(MahoutEvaluator.class);

    Configuration conf = job.getConfiguration();

    FileSystem fs = FileSystem.get(input.toUri(), conf);
    HadoopUtil.delete(conf, input);
    HadoopUtil.delete(conf, output);
    
    storePopulation(fs, new Path(input, "population"), population);

    configureJob(job, conf, evaluator, input, output);
    job.waitForCompletion(true);
    
    OutputUtils.importEvaluations(fs, conf, output, evaluations);
  }

  /**
   * Configure the job
   *
   * @param evaluator
   *          FitnessEvaluator passed to the mapper
   * @param inpath
   *          input {@code Path}
   * @param outpath
   *          output {@code Path}
   */
  private static void configureJob(Job job,
                                   Configuration conf,
                                   FitnessEvaluator<?> evaluator,
                                   Path inpath,
                                   Path outpath) {

    conf.set("mapred.input.dir", inpath.toString());
    conf.set("mapred.output.dir", outpath.toString());

    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
    
    job.setMapperClass(EvalMapper.class);
    
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
    // store the stringified evaluator
    conf.set(EvalMapper.MAHOUT_GA_EVALUATOR, StringUtils.toString(evaluator));
  }
  
  /**
   * Stores a population of candidates in the output file path.
   * 
   * @param fs
   *          FileSystem used to create the output file
   * @param f
   *          output file path
   * @param population
   *          population to store
   */
  static void storePopulation(FileSystem fs, Path f, Iterable<?> population) throws IOException {
    FSDataOutputStream out = fs.create(f);
    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out));
    
    try {
      for (Object candidate : population) {
        writer.write(StringUtils.toString(candidate));
        writer.newLine();
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }
  
}
