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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.SequenceFile.Sorter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.StringUtils;
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
   * @param evaluations <code>List&lt;CDFitness&gt;</code> that contains the
   *        evaluated fitness for each candidate from the input population,
   *        sorted in the same order as the candidates.
   * @param split DatasetSplit used to separate training and testing input
   * @throws IOException
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  public static void evaluate(List<? extends Rule> rules, int target, Path inpath, Path output, List<CDFitness> evaluations,
      DatasetSplit split) throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();

    Job job = new Job(conf);
    FileSystem fs = FileSystem.get(inpath.toUri(), conf);

    // check the input
    if (!fs.exists(inpath) || !fs.getFileStatus(inpath).isDir()) {
      throw new IllegalArgumentException("Input path not found or is not a directory");
    }

    configureJob(job, rules, target, inpath, output, split);
    job.waitForCompletion(true);

    importEvaluations(fs, conf, output, evaluations);
  }

  /**
   * Initializes the dataset
   * 
   * @param inpath input path (the dataset)
   * @throws IOException
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
   * @throws IOException
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  public static CDFitness evaluate(Rule rule, int target, Path inpath, Path output, DatasetSplit split) throws IOException, InterruptedException, ClassNotFoundException {
    List<CDFitness> evals = new ArrayList<CDFitness>();

    evaluate(Arrays.asList(rule), target, inpath, output, evals, split);

    return evals.get(0);
  }

  /**
   * Use all the dataset for training.
   * 
   * @param rules classification rules to evaluate
   * @param target label value to evaluate the rules for
   * @param inpath input path (the dataset)
   * @param evaluations <code>List&lt;CDFitness&gt;</code> that contains the
   *        evaluated fitness for each candidate from the input population,
   *        sorted in the same order as the candidates.
   * @throws IOException
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  public static void evaluate(List<? extends Rule> rules, int target, Path inpath, Path output, List<CDFitness> evaluations)
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
   * @param outpath output <code>Path</code>
   * @param split DatasetSplit used to separate training and testing input
   * @throws IOException 
   */
  private static void configureJob(Job job, List<? extends Rule> rules, int target, Path inpath, Path outpath, DatasetSplit split) throws IOException {
    split.storeJobParameters(job.getConfiguration());

    FileInputFormat.setInputPaths(job, inpath);
    FileOutputFormat.setOutputPath(job, outpath);

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
   * @param outpath output <code>Path</code>
   * @param evaluations <code>List&lt;Fitness&gt;</code> that contains the
   *        evaluated fitness for each candidate from the input population,
   *        sorted in the same order as the candidates.
   * @throws IOException
   */
  private static void importEvaluations(FileSystem fs, Configuration conf, Path outpath, List<CDFitness> evaluations) throws IOException {
    Sorter sorter = new Sorter(fs, LongWritable.class, CDFitness.class, conf);

    // merge and sort the outputs
    Path[] outfiles = OutputUtils.listOutputFiles(fs, outpath);
    Path output = new Path(outpath, "output.sorted");
    sorter.merge(outfiles, output);

    // import the evaluations
    LongWritable key = new LongWritable();
    CDFitness value = new CDFitness();
    Reader reader = new Reader(fs, output, conf);

    while (reader.next(key, value)) {
      evaluations.add(new CDFitness(value));
    }

    reader.close();
  }

}
