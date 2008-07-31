package org.apache.mahout.ga.watchmaker.cd.hadoop;

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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.SequenceFile.Sorter;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.ga.watchmaker.OutputUtils;
import org.apache.mahout.utils.StringUtils;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.ga.watchmaker.cd.DataSet;
import org.apache.mahout.ga.watchmaker.cd.FileInfoParser;
import org.apache.mahout.ga.watchmaker.cd.Rule;
import org.apache.mahout.ga.watchmaker.cd.hadoop.DatasetSplit.DatasetTextInputFormat;

/**
 * Mahout distributed evaluator. takes a list of classification rules and an
 * input path and launch a Hadoop job to evaluate the fitness of each rule. At
 * the end loads the evaluations from the job output.
 * 
 */
public class CDMahoutEvaluator {

  /**
   * Uses Mahout to evaluate the classification rules using the given evaluator.
   * The input path contains the dataset
   * 
   * @param rules classification rules to evaluate
   * @param inpath input path (the dataset)
   * @param evaluations <code>List&lt;Fitness&gt;</code> that contains the
   *        evaluated fitness for each candidate from the input population,
   *        sorted in the same order as the candidates.
   * @param split DatasetSplit used to separate training and testing input
   * @throws IOException
   */
  public static void evaluate(List<? extends Rule> rules, Path inpath,
      List<CDFitness> evaluations, DatasetSplit split) throws IOException {
    JobConf conf = new JobConf(CDMahoutEvaluator.class);
    FileSystem fs = FileSystem.get(conf);

    // check the input
    if (!fs.exists(inpath) || !fs.getFileStatus(inpath).isDir())
      throw new RuntimeException("Input path not found or is not a directory");

    Path outpath = OutputUtils.prepareOutput(fs);

    configureJob(conf, rules, inpath, outpath, split);
    JobClient.runJob(conf);

    importEvaluations(fs, conf, outpath, evaluations);
  }

  public static void InitializeDataSet(Path inpath) throws IOException {
    JobConf conf = new JobConf(CDMahoutEvaluator.class);
    FileSystem fs = FileSystem.get(conf);

    // Initialize the dataset
    DataSet.initialize(FileInfoParser.parseFile(fs, inpath));
  }

  /**
   * Evaluate a single rule.
   * 
   * @param rule
   * @param inpath
   * @param split
   * @return the evaluation
   * @throws IOException
   */
  public static CDFitness evaluate(Rule rule, Path inpath, DatasetSplit split)
      throws IOException {
    List<CDFitness> evals = new ArrayList<CDFitness>();

    evaluate(Arrays.asList(rule), inpath, evals, split);

    return evals.get(0);
  }

  /**
   * Use all the dataset for training.
   * 
   * @param rules
   * @param inpath
   * @param evaluations
   * @throws IOException
   */
  public static void evaluate(List<? extends Rule> rules, Path inpath,
      List<CDFitness> evaluations) throws IOException {
    evaluate(rules, inpath, evaluations, new DatasetSplit(1));
  }

  /**
   * Configure the job
   * 
   * @param conf
   * @param evaluator FitnessEvaluator passed to the mapper
   * @param inpath input <code>Path</code>
   * @param outpath output <code>Path</code>
   */
  private static void configureJob(JobConf conf, List<? extends Rule> rules,
      Path inpath, Path outpath, DatasetSplit split) {
    split.storeJobParameters(conf);

    DatasetTextInputFormat.setInputPaths(conf, inpath);
    SequenceFileOutputFormat.setOutputPath(conf, outpath);

    conf.setOutputKeyClass(LongWritable.class);
    conf.setOutputValueClass(CDFitness.class);

    conf.setMapperClass(CDMapper.class);
    conf.setCombinerClass(CDReducer.class);
    conf.setReducerClass(CDReducer.class);

    conf.setInputFormat(DatasetTextInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    // store the stringified rules
    conf.set(CDMapper.CLASSDISCOVERY_RULES, StringUtils.toString(rules));
  }

  /**
   * Reads back the evaluations.
   * 
   * @param fs
   * @param conf
   * @param outpath output <code>Path</code>
   * @param evaluations List of evaluations
   * @throws IOException
   */
  private static void importEvaluations(FileSystem fs, JobConf conf,
      Path outpath, List<CDFitness> evaluations) throws IOException {
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
