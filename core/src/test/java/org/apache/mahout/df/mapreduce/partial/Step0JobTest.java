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

package org.apache.mahout.df.mapreduce.partial;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import junit.framework.TestCase;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.data.DataConverter;
import org.apache.mahout.df.data.DataLoader;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.data.Utils;
import org.apache.mahout.df.mapreduce.Builder;
import org.apache.mahout.df.mapreduce.partial.Step0Job.Step0Mapper;
import org.apache.mahout.df.mapreduce.partial.Step0Job.Step0Output;

public class Step0JobTest extends TestCase {

  // the generated data must be big enough to be splited by FileInputFormat

  static final int numAttributes = 40;

  static final int numInstances = 2000;

  int numTrees = 10;

  static final int numMaps = 5;

  Step0Context context;

  /**
   * Computes the "mapred.max.split.size" that will generate the desired number
   * of input splits
   * 
   * @param conf
   * @param inputPath
   * @param numMaps desired number of input splits
   * @throws Exception
   */
  public static void setMaxSplitSize(Configuration conf, Path inputPath,
      int numMaps) throws Exception {
    FileSystem fs = inputPath.getFileSystem(conf);
    FileStatus status = fs.getFileStatus(inputPath);
    long goalSize = status.getLen() / numMaps;
    conf.setLong("mapred.max.split.size", goalSize);
  }

  public void testStep0Mapper() throws Exception {
    Random rng = RandomUtils.getRandom();

    // create a dataset large enough to be split up
    String descriptor = Utils.randomDescriptor(rng, numAttributes);
    double[][] source = Utils.randomDoubles(rng, descriptor, numInstances);
    String[] sData = Utils.double2String(source);

    // write the data to a file
    Path dataPath = Utils.writeDataToTestFile(sData);

    Job job = new Job();
    job.setInputFormatClass(TextInputFormat.class);
    FileInputFormat.setInputPaths(job, dataPath);

    setMaxSplitSize(job.getConfiguration(), dataPath, numMaps);

    // retrieve the splits
    TextInputFormat input = new TextInputFormat();
    List<InputSplit> splits = input.getSplits(job);
    assertEquals(numMaps, splits.size());

    InputSplit[] sorted = new InputSplit[numMaps];
    splits.toArray(sorted);
    Builder.sortSplits(sorted);

    context = new Step0Context(new Step0Mapper(), job.getConfiguration(),
        new TaskAttemptID(), numMaps);

    for (int p = 0; p < numMaps; p++) {
      InputSplit split = sorted[p];

      RecordReader<LongWritable, Text> reader = input.createRecordReader(split,
          context);
      reader.initialize(split, context);

      Step0Mapper mapper = new Step0Mapper();
      mapper.configure(p);

      Long firstKey = null;
      int size = 0;

      while (reader.nextKeyValue()) {
        LongWritable key = reader.getCurrentKey();

        if (firstKey == null) {
          firstKey = key.get();
        }

        mapper.map(key, reader.getCurrentValue(), context);

        size++;
      }

      mapper.cleanup(context);

      // validate the mapper's output
      assertEquals(p, context.keys[p]);
      assertEquals(firstKey.longValue(), context.values[p].getFirstId());
      assertEquals(size, context.values[p].getSize());
    }

  }

  public void testProcessOutput() throws Exception {
    Random rng = RandomUtils.getRandom();

    // create a dataset large enough to be split up
    String descriptor = Utils.randomDescriptor(rng, numAttributes);
    double[][] source = Utils.randomDoubles(rng, descriptor, numInstances);

    // each instance label is its index in the dataset
    int labelId = Utils.findLabel(descriptor);
    for (int index = 0; index < numInstances; index++) {
      source[index][labelId] = index;
    }

    String[] sData = Utils.double2String(source);

    // write the data to a file
    Path dataPath = Utils.writeDataToTestFile(sData);

    // prepare a data converter
    Dataset dataset = DataLoader.generateDataset(descriptor, sData);
    DataConverter converter = new DataConverter(dataset);

    Job job = new Job();
    job.setInputFormatClass(TextInputFormat.class);
    FileInputFormat.setInputPaths(job, dataPath);

    setMaxSplitSize(job.getConfiguration(), dataPath, numMaps);

    // retrieve the splits
    TextInputFormat input = new TextInputFormat();
    List<InputSplit> splits = input.getSplits(job);
    assertEquals(numMaps, splits.size());

    InputSplit[] sorted = new InputSplit[numMaps];
    splits.toArray(sorted);
    Builder.sortSplits(sorted);

    List<Integer> keys = new ArrayList<Integer>();
    List<Step0Output> values = new ArrayList<Step0Output>();

    int[] expectedIds = new int[numMaps];

    TaskAttemptContext context = new TaskAttemptContext(job.getConfiguration(),
        new TaskAttemptID());

    for (int p = 0; p < numMaps; p++) {
      InputSplit split = sorted[p];
      RecordReader<LongWritable, Text> reader = input.createRecordReader(split,
          context);
      reader.initialize(split, context);

      Long firstKey = null;
      int size = 0;

      while (reader.nextKeyValue()) {
        LongWritable key = reader.getCurrentKey();
        Text value = reader.getCurrentValue();

        if (firstKey == null) {
          firstKey = key.get();
          expectedIds[p] = converter.convert(0, value.toString()).label;
        }

        size++;
      }

      keys.add(p);
      values.add(new Step0Output(firstKey, size));
    }

    Step0Output[] partitions = Step0Job.processOutput(keys, values);

    int[] actualIds = Step0Output.extractFirstIds(partitions);

    assertTrue("Expected: " + Arrays.toString(expectedIds) + " But was: "
        + Arrays.toString(actualIds), Arrays.equals(expectedIds, actualIds));
  }

  public class Step0Context extends Context {

    private final int[] keys;

    private final Step0Output[] values;

    private int index = 0;

    public Step0Context(Mapper<?,?,?,?> mapper, Configuration conf,
        TaskAttemptID taskid, int numMaps) throws IOException,
        InterruptedException {
      mapper.super(conf, taskid, null, null, null, null, null);

      keys = new int[numMaps];
      values = new Step0Output[numMaps];
    }

    @Override
    public void write(Object key, Object value) throws IOException,
        InterruptedException {
      if (index == keys.length) {
        throw new IOException("Received more output than expected : " + index);
      }

      keys[index] = ((IntWritable) key).get();
      values[index] = ((Step0Output) value).clone();

      index++;
    }

    /**
     * Number of outputs collected
     * 
     * @return
     */
    public int nbOutputs() {
      return index;
    }

    public int[] getKeys() {
      return keys;
    }

    public Step0Output[] getValues() {
      return values;
    }
  }
}
