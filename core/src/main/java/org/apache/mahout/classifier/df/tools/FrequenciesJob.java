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

package org.apache.mahout.classifier.df.tools;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.mapreduce.Builder;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.util.Arrays;

/**
 * Temporary class used to compute the frequency distribution of the "class attribute".<br>
 * This class can be used when the criterion variable is the categorical attribute.
 */
public class FrequenciesJob {
  
  private static final Logger log = LoggerFactory.getLogger(FrequenciesJob.class);
  
  /** directory that will hold this job's output */
  private final Path outputPath;
  
  /** file that contains the serialized dataset */
  private final Path datasetPath;
  
  /** directory that contains the data used in the first step */
  private final Path dataPath;
  
  /**
   * @param base
   *          base directory
   * @param dataPath
   *          data used in the first step
   */
  public FrequenciesJob(Path base, Path dataPath, Path datasetPath) {
    this.outputPath = new Path(base, "frequencies.output");
    this.dataPath = dataPath;
    this.datasetPath = datasetPath;
  }
  
  /**
   * @return counts[partition][label] = num tuples from 'partition' with class == label
   */
  public int[][] run(Configuration conf) throws IOException, ClassNotFoundException, InterruptedException {
    
    // check the output
    FileSystem fs = outputPath.getFileSystem(conf);
    if (fs.exists(outputPath)) {
      throw new IOException("Output path already exists : " + outputPath);
    }
    
    // put the dataset into the DistributedCache
    URI[] files = {datasetPath.toUri()};
    DistributedCache.setCacheFiles(files, conf);
    
    Job job = new Job(conf);
    job.setJarByClass(FrequenciesJob.class);
    
    FileInputFormat.setInputPaths(job, dataPath);
    FileOutputFormat.setOutputPath(job, outputPath);
    
    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(IntWritable.class);
    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(Frequencies.class);
    
    job.setMapperClass(FrequenciesMapper.class);
    job.setReducerClass(FrequenciesReducer.class);
    
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
    // run the job
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
    
    int[][] counts = parseOutput(job);

    HadoopUtil.delete(conf, outputPath);
    
    return counts;
  }
  
  /**
   * Extracts the output and processes it
   * 
   * @return counts[partition][label] = num tuples from 'partition' with class == label
   */
  int[][] parseOutput(JobContext job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    int numMaps = conf.getInt("mapred.map.tasks", -1);
    log.info("mapred.map.tasks = {}", numMaps);
    
    FileSystem fs = outputPath.getFileSystem(conf);
    
    Path[] outfiles = DFUtils.listOutputFiles(fs, outputPath);
    
    Frequencies[] values = new Frequencies[numMaps];
    
    // read all the outputs
    int index = 0;
    for (Path path : outfiles) {
      for (Frequencies value : new SequenceFileValueIterable<Frequencies>(path, conf)) {
        values[index++] = value;
      }
    }
    
    if (index < numMaps) {
      throw new IllegalStateException("number of output Frequencies (" + index
          + ") is lesser than the number of mappers!");
    }
    
    // sort the frequencies using the firstIds
    Arrays.sort(values);
    return Frequencies.extractCounts(values);
  }
  
  /**
   * Outputs the first key and the label of each tuple
   * 
   */
  private static class FrequenciesMapper extends Mapper<LongWritable,Text,LongWritable,IntWritable> {
    
    private LongWritable firstId;
    
    private DataConverter converter;
    private Dataset dataset;
    
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      
      dataset = Builder.loadDataset(conf);
      setup(dataset);
    }
    
    /**
     * Useful when testing
     */
    void setup(Dataset dataset) {
      converter = new DataConverter(dataset);
    }
    
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException,
                                                                     InterruptedException {
      if (firstId == null) {
        firstId = new LongWritable(key.get());
      }
      
      Instance instance = converter.convert(value.toString());
      
      context.write(firstId, new IntWritable((int) dataset.getLabel(instance)));
    }
    
  }
  
  private static class FrequenciesReducer extends Reducer<LongWritable,IntWritable,LongWritable,Frequencies> {
    
    private int nblabels;
    
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      Configuration conf = context.getConfiguration();
      Dataset dataset = Builder.loadDataset(conf);
      setup(dataset.nblabels());
    }
    
    /**
     * Useful when testing
     */
    void setup(int nblabels) {
      this.nblabels = nblabels;
    }
    
    @Override
    protected void reduce(LongWritable key, Iterable<IntWritable> values, Context context)
      throws IOException, InterruptedException {
      int[] counts = new int[nblabels];
      for (IntWritable value : values) {
        counts[value.get()]++;
      }
      
      context.write(key, new Frequencies(key.get(), counts));
    }
  }
  
  /**
   * Output of the job
   * 
   */
  private static class Frequencies implements Writable, Comparable<Frequencies>, Cloneable {
    
    /** first key of the partition used to sort the partitions */
    private long firstId;
    
    /** counts[c] = num tuples from the partition with label == c */
    private int[] counts;
    
    Frequencies() { }
    
    Frequencies(long firstId, int[] counts) {
      this.firstId = firstId;
      this.counts = Arrays.copyOf(counts, counts.length);
    }
    
    @Override
    public void readFields(DataInput in) throws IOException {
      firstId = in.readLong();
      counts = DFUtils.readIntArray(in);
    }
    
    @Override
    public void write(DataOutput out) throws IOException {
      out.writeLong(firstId);
      DFUtils.writeArray(out, counts);
    }
    
    @Override
    public boolean equals(Object other) {
      return other instanceof Frequencies && firstId == ((Frequencies) other).firstId;
    }
    
    @Override
    public int hashCode() {
      return (int) firstId;
    }
    
    @Override
    protected Frequencies clone() {
      return new Frequencies(firstId, counts);
    }
    
    @Override
    public int compareTo(Frequencies obj) {
      if (firstId < obj.firstId) {
        return -1;
      } else if (firstId > obj.firstId) {
        return 1;
      } else {
        return 0;
      }
    }
    
    public static int[][] extractCounts(Frequencies[] partitions) {
      int[][] counts = new int[partitions.length][];
      for (int p = 0; p < partitions.length; p++) {
        counts[p] = partitions[p].counts;
      }
      return counts;
    }
  }
}
