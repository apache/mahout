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

package org.apache.mahout.classifier.naivebayes.training;

import java.util.Map;

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.VectorWritable;

/** This class trains a Naive Bayes Classifier (Parameters for both Naive Bayes and Complementary Naive Bayes) */
public final class TrainNaiveBayesJob extends AbstractJob {

  public static final String WEIGHTS_PER_FEATURE = "__SPF";
  public static final String WEIGHTS_PER_LABEL = "__SPL";
  public static final String LABEL_THETA_NORMALIZER = "_LTN";

  public static final String SUMMED_OBSERVATIONS = "summedObservations";
  public static final String WEIGHTS = "weights";
  public static final String THETAS = "thetas";

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("labels", "l", "comma-separated list of labels to include in training", true);
    addOption("alphaI", "a", "smoothing parameter", String.valueOf(1.0f));
    addOption("trainComplementary", "c", "train complementary?", String.valueOf(false));

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Iterable<String> labels = Splitter.on(",").split(parsedArgs.get("--labels"));
    float alphaI = Float.parseFloat(parsedArgs.get("--alphaI"));
    boolean trainComplementary = Boolean.parseBoolean(parsedArgs.get("--trainComplementary"));

    TrainUtils.writeLabelIndex(getConf(), labels, getTempPath("labelIndex"));
    TrainUtils.setSerializations(getConf());
    TrainUtils.cacheFiles(getTempPath("labelIndex"), getConf());

    Job indexInstances = prepareJob(getInputPath(), getTempPath(SUMMED_OBSERVATIONS), SequenceFileInputFormat.class,
        IndexInstancesMapper.class, IntWritable.class, VectorWritable.class, VectorSumReducer.class, IntWritable.class,
        VectorWritable.class, SequenceFileOutputFormat.class);
    indexInstances.setCombinerClass(VectorSumReducer.class);
    indexInstances.waitForCompletion(true);

    Job weightSummer = prepareJob(getTempPath(SUMMED_OBSERVATIONS), getTempPath(WEIGHTS),
        SequenceFileInputFormat.class, WeightsMapper.class, Text.class, VectorWritable.class, VectorSumReducer.class,
        Text.class, VectorWritable.class, SequenceFileOutputFormat.class);
    weightSummer.getConfiguration().set(WeightsMapper.NUM_LABELS, String.valueOf(Iterables.size(labels)));
    weightSummer.setCombinerClass(VectorSumReducer.class);
    weightSummer.waitForCompletion(true);

    TrainUtils.cacheFiles(getTempPath(WEIGHTS), getConf());

    Job thetaSummer = prepareJob(getTempPath(SUMMED_OBSERVATIONS), getTempPath(THETAS),
        SequenceFileInputFormat.class, ThetaMapper.class, Text.class, VectorWritable.class, VectorSumReducer.class,
        Text.class, VectorWritable.class, SequenceFileOutputFormat.class);
    thetaSummer.setCombinerClass(VectorSumReducer.class);
    thetaSummer.getConfiguration().setFloat(ThetaMapper.ALPHA_I, alphaI);
    thetaSummer.getConfiguration().setBoolean(ThetaMapper.TRAIN_COMPLEMENTARY, trainComplementary);
    thetaSummer.waitForCompletion(true);

    NaiveBayesModel naiveBayesModel = TrainUtils.readModelFromTempDir(getTempPath(), getConf());
    naiveBayesModel.validate();
    naiveBayesModel.serialize(getOutputPath(), getConf());

    return 0;
  }

}
