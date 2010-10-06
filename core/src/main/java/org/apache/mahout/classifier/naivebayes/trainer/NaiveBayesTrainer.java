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

package org.apache.mahout.classifier.naivebayes.trainer;

import java.io.IOException;
import java.net.URI;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.VectorWritable;

/**
 * This class trains a Naive Bayes Classifier (Parameters for both Naive Bayes and Complementary Naive Bayes)
 * 
 * 
 */
public final class NaiveBayesTrainer {
  
  public static final String THETA_SUM = "thetaSum";
  public static final String SUM_VECTORS = "sumVectors";
  public static final String CLASS_VECTORS = "classVectors";
  public static final String LABEL_MAP = "labelMap";
  public static final String ALPHA_I = "alphaI";

  public static void trainNaiveBayes(Path input,
                                      Configuration conf,
                                      List<String> inputLabels,
                                      Path output,
                                      int numReducers,
                                      float alphaI,
                                      boolean trainComplementary)
      throws IOException, InterruptedException, ClassNotFoundException {
    conf.setFloat(ALPHA_I, alphaI);
    Path labelMapPath = createLabelMapFile(inputLabels, conf, new Path(output, LABEL_MAP));
    Path classVectorPath =  new Path(output, CLASS_VECTORS);
    runNaiveBayesByLabelSummer(input, conf, labelMapPath, classVectorPath, numReducers);
    Path weightFilePath = new Path(output, SUM_VECTORS);
    runNaiveBayesWeightSummer(classVectorPath, conf, labelMapPath, weightFilePath, numReducers);
    Path thetaFilePath = new Path(output, THETA_SUM);
    if (trainComplementary) {
      runNaiveBayesThetaComplementarySummer(classVectorPath, conf, weightFilePath, thetaFilePath, numReducers);
    } else {
      runNaiveBayesThetaSummer(classVectorPath, conf, weightFilePath, thetaFilePath, numReducers);
    }
  }

  private static void runNaiveBayesByLabelSummer(Path input, Configuration conf, Path labelMapPath,
                                                 Path output, int numReducers)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
    DistributedCache.setCacheFiles(new URI[] {labelMapPath.toUri()}, conf);
  
    Job job = new Job(conf);
    job.setJobName("Train Naive Bayes: input-folder: " + input + ", label-map-file: "
        + labelMapPath.toString());
    job.setJarByClass(NaiveBayesTrainer.class);
    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);
    job.setMapperClass(NaiveBayesInstanceMapper.class);
    job.setCombinerClass(NaiveBayesSumReducer.class);
    job.setReducerClass(NaiveBayesSumReducer.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setNumReduceTasks(numReducers);
    HadoopUtil.overwriteOutput(output);
    job.waitForCompletion(true);
  }

  private static void runNaiveBayesWeightSummer(Path input, Configuration conf,
                                                Path labelMapPath, Path output, int numReducers)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
    DistributedCache.setCacheFiles(new URI[] {labelMapPath.toUri()}, conf);
    
    Job job = new Job(conf);
    job.setJobName("Train Naive Bayes: input-folder: " + input);
    job.setJarByClass(NaiveBayesTrainer.class);
    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);
    job.setMapperClass(NaiveBayesWeightsMapper.class);
    job.setReducerClass(NaiveBayesSumReducer.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setNumReduceTasks(numReducers);
    HadoopUtil.overwriteOutput(output);
    job.waitForCompletion(true);
  }
  
  private static void runNaiveBayesThetaSummer(Path input, Configuration conf,
                                               Path weightFilePath, Path output, int numReducers)
      throws IOException, InterruptedException, ClassNotFoundException {
    
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
    DistributedCache.setCacheFiles(new URI[] {weightFilePath.toUri()}, conf);
  
    Job job = new Job(conf);
    job.setJobName("Train Naive Bayes: input-folder: " + input + ", label-map-file: "
        + weightFilePath.toString());
    job.setJarByClass(NaiveBayesTrainer.class);
    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);
    job.setMapperClass(NaiveBayesThetaMapper.class);
    job.setReducerClass(NaiveBayesSumReducer.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setNumReduceTasks(numReducers);
    HadoopUtil.overwriteOutput(output);
    job.waitForCompletion(true);
  }

  private static void runNaiveBayesThetaComplementarySummer(Path input, Configuration conf,
                                                            Path weightFilePath, Path output, int numReducers)
      throws IOException, InterruptedException, ClassNotFoundException {
    
    // this conf parameter needs to be set enable serialisation of conf values
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
    DistributedCache.setCacheFiles(new URI[] {weightFilePath.toUri()}, conf);
  
    Job job = new Job(conf);
    job.setJobName("Train Naive Bayes: input-folder: " + input + ", label-map-file: "
        + weightFilePath.toString());
    job.setJarByClass(NaiveBayesTrainer.class);
    FileInputFormat.setInputPaths(job, input);
    FileOutputFormat.setOutputPath(job, output);
    job.setMapperClass(NaiveBayesThetaComplementaryMapper.class);
    job.setReducerClass(NaiveBayesSumReducer.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setNumReduceTasks(numReducers);
    HadoopUtil.overwriteOutput(output);
    job.waitForCompletion(true);
  }

  
  
  /**
   * Write the list of labels into a map file
   * 
   * @param wordCountPath
   * @param dictionaryPathBase
   * @throws IOException
   */
  public static Path createLabelMapFile(List<String> labels,
                                         Configuration conf,
                                         Path labelMapPathBase) throws IOException {
    FileSystem fs = FileSystem.get(labelMapPathBase.toUri(), conf);
    Path labelMapPath = new Path(labelMapPathBase, LABEL_MAP);
    
    SequenceFile.Writer dictWriter = new SequenceFile.Writer(fs, conf, labelMapPath, Text.class, IntWritable.class);
    int i = 0;
    for (String label : labels) {
      Writable key = new Text(label);
      dictWriter.append(key, new IntWritable(i++));
    }
    return labelMapPath;
  }
}
