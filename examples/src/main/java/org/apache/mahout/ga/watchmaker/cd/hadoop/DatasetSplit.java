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

package org.apache.mahout.ga.watchmaker.cd.hadoop;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.LineRecordReader;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.mahout.utils.StringUtils;
import org.uncommons.maths.random.MersenneTwisterRNG;

import java.io.IOException;
import java.util.Random;

/**
 * Separate the input data into a training and testing set.
 */
public class DatasetSplit {

  private static final String SEED = "traintest.seed";

  private static final String THRESHOLD = "traintest.threshold";

  private static final String TRAINING = "traintest.training";

  private final byte[] seed;

  private final double threshold;

  private boolean training;

  /**
   * 
   * @param seed
   * @param threshold fraction of the total dataset that will be used for
   *        training
   */
  public DatasetSplit(byte[] seed, double threshold) {
    this.seed = seed;
    this.threshold = threshold;
    this.training = true;
  }

  public DatasetSplit(double threshold) {
    this(new MersenneTwisterRNG().getSeed(), threshold);
  }

  public DatasetSplit(JobConf conf) {
    seed = getSeed(conf);
    threshold = getThreshold(conf);
    training = isTraining(conf);
  }

  public byte[] getSeed() {
    return seed;
  }

  public double getThreshold() {
    return threshold;
  }

  public boolean isTraining() {
    return training;
  }

  public void setTraining(boolean training) {
    this.training = training;
  }

  public void storeJobParameters(JobConf conf) {
    conf.set(SEED, StringUtils.toString(seed));
    conf.set(THRESHOLD, Double.toString(threshold));
    conf.setBoolean(TRAINING, training);
  }

  static byte[] getSeed(JobConf conf) {
    String seedstr = conf.get(SEED);
    if (seedstr == null)
      throw new RuntimeException("SEED job parameter not found");

    return (byte[]) StringUtils.fromString(seedstr);
  }

  static double getThreshold(JobConf conf) {
    String thrstr = conf.get(THRESHOLD);
    if (thrstr == null)
      throw new RuntimeException("THRESHOLD job parameter not found");

    return Double.parseDouble(thrstr);
  }

  static boolean isTraining(JobConf conf) {
    if (conf.get(TRAINING) == null)
      throw new RuntimeException("TRAINING job parameter not found");

    return conf.getBoolean(TRAINING, true);
  }

  /**
   * a {@link org.apache.hadoop.mapred.LineRecordReader LineRecordReader} that
   * skips some lines from the input. Uses a Random number generator with a
   * specific seed to decide if a line will be skipped or not.
   */
  public static class RndLineRecordReader implements
      RecordReader<LongWritable, Text> {

    private RecordReader<LongWritable, Text> reader;

    private Random rng;

    private double threshold;

    private boolean training;

    private final LongWritable k = new LongWritable();

    private final Text v = new Text();

    public RndLineRecordReader(RecordReader<LongWritable, Text> reader,
        JobConf conf) {
      assert reader != null : "null reader";

      this.reader = reader;

      DatasetSplit split = new DatasetSplit(conf);

      rng = new MersenneTwisterRNG(split.getSeed());
      threshold = split.getThreshold();
      training = split.isTraining();
    }

    @Override
    public void close() throws IOException {
      reader.close();
    }

    @Override
    public LongWritable createKey() {
      return reader.createKey();
    }

    @Override
    public Text createValue() {
      return reader.createValue();
    }

    @Override
    public long getPos() throws IOException {
      return reader.getPos();
    }

    @Override
    public float getProgress() throws IOException {
      return reader.getProgress();
    }

    @Override
    public boolean next(LongWritable key, Text value) throws IOException {
      boolean read;
      do {
        read = reader.next(k, v);
      } while (read && !selected());

      if (!read)
        return false;

      key.set(k.get());
      value.set(v);
      return true;
    }

    /**
     * 
     * @return true if the current input line is not skipped.
     */
    private boolean selected() {
      return training ? rng.nextDouble() < threshold
          : rng.nextDouble() >= threshold;
    }
  }

  /**
   * {@link org.apache.hadoop.mapred.TextInputFormat TextInputFormat that uses a {@link RndLineRecordReader RndLineRecordReader}}
   * as a RecordReader
   */
  public static class DatasetTextInputFormat extends TextInputFormat {

    @Override
    public RecordReader<LongWritable, Text> getRecordReader(InputSplit split,
        JobConf job, Reporter reporter) throws IOException {
      reporter.setStatus(split.toString());

      return new RndLineRecordReader(new LineRecordReader(job,
          (FileSplit) split), job);
    }
  }
}
