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

package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.RandomUtils;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * <p>Split a recommendation dataset into a training and a test set</p>
 *
  * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--input (path): Directory containing one or more text files with the dataset</li>
 * <li>--output (path): path where output should go</li>
 * <li>--trainingPercentage (double): percentage of the data to use as training set (optional, default 0.9)</li>
 * <li>--probePercentage (double): percentage of the data to use as probe set (optional, default 0.1)</li>
 * </ol>
 */
public class DatasetSplitter extends AbstractJob {

  private static final String TRAINING_PERCENTAGE = DatasetSplitter.class.getName() + ".trainingPercentage";
  private static final String PROBE_PERCENTAGE = DatasetSplitter.class.getName() + ".probePercentage";
  private static final String PART_TO_USE = DatasetSplitter.class.getName() + ".partToUse";

  private static final Text INTO_TRAINING_SET = new Text("T");
  private static final Text INTO_PROBE_SET = new Text("P");

  private static final double DEFAULT_TRAINING_PERCENTAGE = 0.9;
  private static final double DEFAULT_PROBE_PERCENTAGE = 0.1;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new DatasetSplitter(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("trainingPercentage", "t", "percentage of the data to use as training set (default: " 
        + DEFAULT_TRAINING_PERCENTAGE + ')', String.valueOf(DEFAULT_TRAINING_PERCENTAGE));
    addOption("probePercentage", "p", "percentage of the data to use as probe set (default: " 
        + DEFAULT_PROBE_PERCENTAGE + ')', String.valueOf(DEFAULT_PROBE_PERCENTAGE));

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    double trainingPercentage = Double.parseDouble(getOption("trainingPercentage"));
    double probePercentage = Double.parseDouble(getOption("probePercentage"));
    String tempDir = getOption("tempDir");

    Path markedPrefs = new Path(tempDir, "markedPreferences");
    Path trainingSetPath = new Path(getOutputPath(), "trainingSet");
    Path probeSetPath = new Path(getOutputPath(), "probeSet");

    Job markPreferences = prepareJob(getInputPath(), markedPrefs, TextInputFormat.class, MarkPreferencesMapper.class,
        Text.class, Text.class, SequenceFileOutputFormat.class);
    markPreferences.getConfiguration().set(TRAINING_PERCENTAGE, String.valueOf(trainingPercentage));
    markPreferences.getConfiguration().set(PROBE_PERCENTAGE, String.valueOf(probePercentage));
    boolean succeeded = markPreferences.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    Job createTrainingSet = prepareJob(markedPrefs, trainingSetPath, SequenceFileInputFormat.class,
        WritePrefsMapper.class, NullWritable.class, Text.class, TextOutputFormat.class);
    createTrainingSet.getConfiguration().set(PART_TO_USE, INTO_TRAINING_SET.toString());
    succeeded = createTrainingSet.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    Job createProbeSet = prepareJob(markedPrefs, probeSetPath, SequenceFileInputFormat.class,
        WritePrefsMapper.class, NullWritable.class, Text.class, TextOutputFormat.class);
    createProbeSet.getConfiguration().set(PART_TO_USE, INTO_PROBE_SET.toString());
    succeeded = createProbeSet.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    return 0;
  }

  static class MarkPreferencesMapper extends Mapper<LongWritable,Text,Text,Text> {

    private Random random;
    private double trainingBound;
    private double probeBound;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      random = RandomUtils.getRandom();
      trainingBound = Double.parseDouble(ctx.getConfiguration().get(TRAINING_PERCENTAGE));
      probeBound = trainingBound + Double.parseDouble(ctx.getConfiguration().get(PROBE_PERCENTAGE));
    }

    @Override
    protected void map(LongWritable key, Text text, Context ctx) throws IOException, InterruptedException {
      double randomValue = random.nextDouble();
      if (randomValue <= trainingBound) {
        ctx.write(INTO_TRAINING_SET, text);
      } else if (randomValue <= probeBound) {
        ctx.write(INTO_PROBE_SET, text);
      }
    }
  }

  static class WritePrefsMapper extends Mapper<Text,Text,NullWritable,Text> {

    private String partToUse;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      partToUse = ctx.getConfiguration().get(PART_TO_USE);
    }

    @Override
    protected void map(Text key, Text text, Context ctx) throws IOException, InterruptedException {
      if (partToUse.equals(key.toString())) {
        ctx.write(NullWritable.get(), text);
      }
    }
  }
}
