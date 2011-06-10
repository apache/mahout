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

package org.apache.mahout.ga.watchmaker.cd.hadoop;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.StringUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.ga.watchmaker.OutputUtils;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.ga.watchmaker.cd.DataSet;
import org.apache.mahout.ga.watchmaker.cd.FileInfoParser;
import org.apache.mahout.ga.watchmaker.cd.Rule;
import org.apache.mahout.ga.watchmaker.cd.hadoop.DatasetSplit.DatasetTextInputFormat;

/**
 * Mahout distributed evaluator. takes a list of classification rules and an
 * input path and launch a Hadoop job to evaluate the fitness of each rule. At
 * the end loads the evaluations from the job output.
 */
public final class CDMahoutEvaluator {

  private CDMahoutEvaluator() {
  }

  /**
   * Uses Mahout to evaluate the classification rules using the given evaluator.
   * The input path contains the dataset
   *
   * @param rules classification rules to evaluate
   * @param target label value to evaluate the rules for
   * @param inpath input path (the dataset)
   * @param evaluations {@code List<CDFitness>} that contains the
   *        evaluated fitness for each candidate from the input population,
   *        sorted in the same order as the candidates.
   * @param split DatasetSplit used to separate training and testing input
   */
  public static void evaluate(List<? extends Rule> rules,
                              int target,
                              Path inpath,
                              Path output,
                              Collection<CDFitness> evaluations,
                              DatasetSplit split) throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(inpath.toUri(), conf);
    Preconditions.checkArgument(fs.exists(inpath) && fs.getFileStatus(inpath).isDir(), "%s is not a directory", inpath);

    Job job = new Job(conf);

    configureJob(job, rules, target, inpath, output, split);
    job.waitForCompletion(true);

    importEvaluations(fs, conf, output, evaluations);
  }

  /**
   * Initializes the dataset
   * 
   * @param inpath input path (the dataset)
   */
  public static void initializeDataSet(Path inpath) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(inpath.toUri(), conf);

    // Initialize the dataset
    DataSet.initialize(FileInfoParser.parseFile(fs, inpath));
  }

  /**
   * Evaluate a single rule.
   * 
   * @param rule classification rule to evaluate
   * @param target label value to evaluate the rules for
   * @param inpath input path (the dataset)
   * @param split DatasetSplit used to separate training and testing input
   * @return the evaluation
   */
  public static CDFitness evaluate(Rule rule, int target, Path inpath, Path output, DatasetSplit split)
    throws IOException, InterruptedException, ClassNotFoundException {
    List<CDFitness> evals = Lists.newArrayList();

    evaluate(Arrays.asList(rule), target, inpath, output, evals, split);

    return evals.get(0);
  }

  /**
   * Use all the dataset for training.
   *
   * @param rules classification rules to evaluate
   * @param target label value to evaluate the rules for
   * @param inpath input path (the dataset)
   * @param evaluations {@code List<CDFitness>} that contains the
   *        evaluated fitness for each candidate from the input population,
   *        sorted in the same order as the candidates.
   */
  public static void evaluate(List<? extends Rule> rules,
                              int target,
                              Path inpath,
                              Path output,
                              Collection<CDFitness> evaluations)
    throws IOException, InterruptedException, ClassNotFoundException {
    evaluate(rules, target, inpath, output, evaluations, new DatasetSplit(1));
  }

  /**
   * Configure the job
   *
   * @param job Job to configure
   * @param rules classification rules to evaluate
   * @param target label value to evaluate the rules for
   * @param inpath input path (the dataset)
   * @param outpath output {@code Path}
   * @param split DatasetSplit used to separate training and testing input
   */
  private static void configureJob(Job job,
                                   List<? extends Rule> rules,
                                   int target,
                                   Path inpath,
                                   Path outpath,
                                   DatasetSplit split) throws IOException {
    split.storeJobParameters(job.getConfiguration());

    FileInputFormat.setInputPaths(job, inpath);
    FileOutputFormat.setOutputPath(job, outpath);

    job.setJarByClass(CDMahoutEvaluator.class);
    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(CDFitness.class);

    job.setMapperClass(CDMapper.class);
    job.setCombinerClass(CDReducer.class);
    job.setReducerClass(CDReducer.class);

    job.setInputFormatClass(DatasetTextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    // store the parameters
    Configuration conf = job.getConfiguration();
    conf.set(CDMapper.CLASSDISCOVERY_RULES, StringUtils.toString(rules));
    conf.set(CDMapper.CLASSDISCOVERY_DATASET, StringUtils.toString(DataSet.getDataSet()));
    conf.setInt(CDMapper.CLASSDISCOVERY_TARGET_LABEL, target);
  }

  /**
   * Reads back the evaluations.
   *
   * @param fs File System
   * @param conf Job configuration
   * @param outpath output {@code Path}
   * @param evaluations {@code List<Fitness>} that contains the
   *        evaluated fitness for each candidate from the input population,
   *        sorted in the same order as the candidates.
   */
  private static void importEvaluations(FileSystem fs,
                                        Configuration conf, Path outpath,
                                        Collection<CDFitness> evaluations)
    throws IOException {
    SequenceFile.Sorter sorter = new SequenceFile.Sorter(fs, LongWritable.class, CDFitness.class, conf);

    // merge and sort the outputs
    Path[] outfiles = OutputUtils.listOutputFiles(fs, outpath);
    Path output = new Path(outpath, "output.sorted");
    sorter.merge(outfiles, output);

    // import the evaluations
    for (CDFitness value : new SequenceFileValueIterable<CDFitness>(output, conf)) {
      evaluations.add(value);
    }

  }

}
