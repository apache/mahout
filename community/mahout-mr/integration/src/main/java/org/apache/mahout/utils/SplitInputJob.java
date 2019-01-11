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

package org.apache.mahout.utils;

import java.io.IOException;
import java.io.Serializable;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterator;

/**
 * Class which implements a map reduce version of SplitInput.
 * This class takes a SequenceFile input, e.g. a set of training data
 * for a learning algorithm, downsamples it, applies a random
 * permutation and splits it into test and training sets
 */
public final class SplitInputJob {

  private static final String DOWNSAMPLING_FACTOR = "SplitInputJob.downsamplingFactor";
  private static final String RANDOM_SELECTION_PCT = "SplitInputJob.randomSelectionPct";
  private static final String TRAINING_TAG = "training";
  private static final String TEST_TAG = "test";

  private SplitInputJob() {}

  /**
   * Run job to downsample, randomly permute and split data into test and
   * training sets. This job takes a SequenceFile as input and outputs two
   * SequenceFiles test-r-00000 and training-r-00000 which contain the test and
   * training sets respectively
   *
   * @param initialConf
   *          Initial configuration
   * @param inputPath
   *          path to input data SequenceFile
   * @param outputPath
   *          path for output data SequenceFiles
   * @param keepPct
   *          percentage of key value pairs in input to keep. The rest are
   *          discarded
   * @param randomSelectionPercent
   *          percentage of key value pairs to allocate to test set. Remainder
   *          are allocated to training set
   */
  @SuppressWarnings("rawtypes")
  public static void run(Configuration initialConf, Path inputPath,
      Path outputPath, int keepPct, float randomSelectionPercent)
    throws IOException, ClassNotFoundException, InterruptedException {

    int downsamplingFactor = (int) (100.0 / keepPct);
    initialConf.setInt(DOWNSAMPLING_FACTOR, downsamplingFactor);
    initialConf.setFloat(RANDOM_SELECTION_PCT, randomSelectionPercent);

    // Determine class of keys and values
    FileSystem fs = FileSystem.get(initialConf);

    SequenceFileDirIterator<? extends WritableComparable, Writable> iterator =
        new SequenceFileDirIterator<>(inputPath,
            PathType.LIST, PathFilters.partFilter(), null, false, fs.getConf());
    Class<? extends WritableComparable> keyClass;
    Class<? extends Writable> valueClass;
    if (iterator.hasNext()) {
      Pair<? extends WritableComparable, Writable> pair = iterator.next();
      keyClass = pair.getFirst().getClass();
      valueClass = pair.getSecond().getClass();
    } else {
      throw new IllegalStateException("Couldn't determine class of the input values");
    }

    Job job = new Job(new Configuration(initialConf));

    MultipleOutputs.addNamedOutput(job, TRAINING_TAG, SequenceFileOutputFormat.class, keyClass, valueClass);
    MultipleOutputs.addNamedOutput(job, TEST_TAG, SequenceFileOutputFormat.class, keyClass, valueClass);
    job.setJarByClass(SplitInputJob.class);
    FileInputFormat.addInputPath(job, inputPath);
    FileOutputFormat.setOutputPath(job, outputPath);
    job.setNumReduceTasks(1);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(SplitInputMapper.class);
    job.setReducerClass(SplitInputReducer.class);
    job.setSortComparatorClass(SplitInputComparator.class);
    job.setOutputKeyClass(keyClass);
    job.setOutputValueClass(valueClass);
    job.submit();
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }

  /** Mapper which downsamples the input by downsamplingFactor */
  public static class SplitInputMapper extends
      Mapper<WritableComparable<?>, Writable, WritableComparable<?>, Writable> {

    private int downsamplingFactor;

    @Override
    public void setup(Context ctx) {
      downsamplingFactor = ctx.getConfiguration().getInt(DOWNSAMPLING_FACTOR, 1);
    }

    /** Only run map() for one out of every downsampleFactor inputs */
    @Override
    public void run(Context context) throws IOException, InterruptedException {
      setup(context);
      int i = 0;
      while (context.nextKeyValue()) {
        if (i % downsamplingFactor == 0) {
          map(context.getCurrentKey(), context.getCurrentValue(), context);
        }
        i++;
      }
      cleanup(context);
    }

  }

  /** Reducer which uses MultipleOutputs to randomly allocate key value pairs between test and training outputs */
  public static class SplitInputReducer extends
      Reducer<WritableComparable<?>, Writable, WritableComparable<?>, Writable> {

    private MultipleOutputs multipleOutputs;
    private final Random rnd = RandomUtils.getRandom();
    private float randomSelectionPercent;

    @Override
    protected void setup(Context ctx) throws IOException {
      randomSelectionPercent = ctx.getConfiguration().getFloat(RANDOM_SELECTION_PCT, 0);
      multipleOutputs = new MultipleOutputs(ctx);
    }

    /**
     * Randomly allocate key value pairs between test and training sets.
     * randomSelectionPercent of the pairs will go to the test set.
     */
    @Override
    protected void reduce(WritableComparable<?> key, Iterable<Writable> values,
        Context context) throws IOException, InterruptedException {
      for (Writable value : values) {
        if (rnd.nextInt(100) < randomSelectionPercent) {
          multipleOutputs.write(TEST_TAG, key, value);
        } else {
          multipleOutputs.write(TRAINING_TAG, key, value);
        }
      }

    }

    @Override
    protected void cleanup(Context context) throws IOException {
      try {
        multipleOutputs.close();
      } catch (InterruptedException e) {
        throw new IOException(e);
      }
    }

  }

  /** Randomly permute key value pairs */
  public static class SplitInputComparator extends WritableComparator implements Serializable {

    private final Random rnd = RandomUtils.getRandom();

    protected SplitInputComparator() {
      super(WritableComparable.class);
    }

    @Override
    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      if (rnd.nextBoolean()) {
        return 1;
      } else {
        return -1;
      }
    }
  }

}
