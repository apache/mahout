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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.util.Arrays;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * preparation step of the partial mapreduce builder. Computes some stats that will be used by the builder.
 */
public class Step0Job {
  
  private static final Logger log = LoggerFactory.getLogger(Step0Job.class);
  
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
   * @param datasetPath
   */
  public Step0Job(Path base, Path dataPath, Path datasetPath) {
    this.outputPath = new Path(base, "step0.output");
    this.dataPath = dataPath;
    this.datasetPath = datasetPath;
  }
  
  /**
   * Computes the partitions' info in Hadoop's order
   * 
   * @param conf
   *          configuration
   * @return partitions' info in Hadoop's order
   */
  public Step0Output[] run(Configuration conf) throws IOException,
                                              ClassNotFoundException,
                                              InterruptedException {
    
    // check the output
    if (outputPath.getFileSystem(conf).exists(outputPath)) {
      throw new IOException("Output path already exists : " + outputPath);
    }
    
    // put the dataset into the DistributedCache
    // use setCacheFiles() to overwrite the first-step cache files
    URI[] files = {datasetPath.toUri()};
    DistributedCache.setCacheFiles(files, conf);
    
    Job job = new Job(conf);
    job.setJarByClass(Step0Job.class);
    
    FileInputFormat.setInputPaths(job, dataPath);
    FileOutputFormat.setOutputPath(job, outputPath);
    
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(Step0Output.class);
    
    job.setMapperClass(Step0Mapper.class);
    job.setNumReduceTasks(0); // no reducers
    
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
    // run the job
    job.waitForCompletion(true);
    
    return parseOutput(job);
  }
  
  /**
   * Extracts the output and processes it
   * 
   * @return info for each partition in Hadoop's order
   */
  protected Step0Output[] parseOutput(JobContext job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    log.info("mapred.map.tasks = {}", conf.getInt("mapred.map.tasks", -1));

    List<Integer> keys = Lists.newArrayList();
    List<Step0Output> values = Lists.newArrayList();

    // read all the outputs
    for (Pair<IntWritable,Step0Output> record
         : new SequenceFileDirIterable<IntWritable,Step0Output>(outputPath,
                                                                PathType.LIST,
                                                                PathFilters.logsCRCFilter(), 
                                                                conf)) {
      keys.add(record.getFirst().get());
      values.add(record.getSecond());
    }

    return processOutput(keys, values);
  }
  
  /**
   * Replaces the first id for each partition in Hadoop's order
   */
  protected static Step0Output[] processOutput(List<Integer> keys, List<Step0Output> values) {
    int numMaps = values.size();
    
    // sort the values using firstId
    Step0Output[] sorted = new Step0Output[numMaps];
    values.toArray(sorted);
    Arrays.sort(sorted);
    
    // compute the partitions firstIds (file order)
    int[] orderedIds = new int[numMaps];
    orderedIds[0] = 0;
    for (int p = 1; p < numMaps; p++) {
      orderedIds[p] = orderedIds[p - 1] + sorted[p - 1].size;
    }
    
    // update the values' first ids
    for (Step0Output value : values) {
      int order = ArrayUtils.indexOf(sorted, value);
      value.firstId = orderedIds[order];
    }
    
    // reorder the values in hadoop's order
    Step0Output[] reordered = new Step0Output[numMaps];
    for (int p = 0; p < numMaps; p++) {
      reordered[keys.get(p)] = values.get(p);
    }
    
    return reordered;
  }
  
  /**
   * Outputs the first key and the size of the partition
   * 
   */
  static class Step0Mapper extends Mapper<LongWritable,Text,IntWritable,Step0Output> {
    
    private int partition;
    
    private int size;
    
    private Long firstId;
    
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      configure(context.getConfiguration().getInt("mapred.task.partition", -1));
    }
    
    /**
     * Useful when testing
     * 
     * @param p
     */
    protected void configure(int p) {
      partition = p;
      Preconditions.checkArgument(partition >= 0, "Wrong partition id: %d", partition);
    }
    
    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException,
                                                                     InterruptedException {
      if (firstId == null) {
        firstId = key.get();
      }
      
      size++;
    }
    
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
      context.write(new IntWritable(partition), new Step0Output(firstId, size));
    }
    
  }
  
  /**
   * Output of the step0's mappers
   * 
   */
  public static class Step0Output implements Writable, Comparable<Step0Output>, Cloneable {
    
    /**
     * first key of the partition<br>
     * used to sort the partition
     */
    private long firstId;
    
    /** number of instances in the partition */
    private int size;

    protected Step0Output() { }
    
    protected Step0Output(long firstId, int size) {
      this.firstId = firstId;
      this.size = size;
    }
    
    protected long getFirstId() {
      return firstId;
    }
    
    protected int getSize() {
      return size;
    }
    
    @Override
    public void readFields(DataInput in) throws IOException {
      firstId = in.readLong();
      size = in.readInt();
    }
    
    @Override
    public void write(DataOutput out) throws IOException {
      out.writeLong(firstId);
      out.writeInt(size);
    }
    
    @Override
    public boolean equals(Object other) {
      return other instanceof Step0Output && firstId == ((Step0Output) other).firstId;
    }
    
    @Override
    public int hashCode() {
      return (int) firstId;
    }
    
    @Override
    protected Step0Output clone() {
      return new Step0Output(firstId, size);
    }
    
    @Override
    public int compareTo(Step0Output obj) {
      if (firstId < obj.firstId) {
        return -1;
      } else if (firstId > obj.firstId) {
        return 1;
      } else {
        return 0;
      }
    }
    
    public static int[] extractFirstIds(Step0Output[] partitions) {
      int[] ids = new int[partitions.length];
      
      for (int p = 0; p < partitions.length; p++) {
        ids[p] = (int) partitions[p].firstId;
      }
      
      return ids;
    }
    
    public static int[] extractSizes(Step0Output[] partitions) {
      int[] sizes = new int[partitions.length];
      
      for (int p = 0; p < partitions.length; p++) {
        sizes[p] = partitions[p].size;
      }
      
      return sizes;
    }
  }
}
