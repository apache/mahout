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

package org.apache.mahout.df.mapred.partial;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.util.Arrays;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.mahout.df.DFUtils;

/**
 * preparation step of the partial mapreduce builder. Computes some stats that
 * will be used by the builder.
 */
public class Step0Job implements Cloneable {

  /** directory that will hold this job's output */
  private final Path outputPath;

  /** file that contains the serialized dataset */
  private final Path datasetPath;

  /** directory that contains the data used in the first step */
  private final Path dataPath;

  /**
   * @param base base directory
   * @param dataPath data used in the first step
   * @param datasetPath
   */
  public Step0Job(Path base, Path dataPath, Path datasetPath) {
    this.outputPath = new Path(base, "step0.output");
    this.dataPath = dataPath;
    this.datasetPath = datasetPath;
  }

  /**
   * Computes the partitions' first ids in Hadoop's order
   * 
   * @param conf configuration
   * @return first ids for all the partitions
   * @throws IOException
   */
  public Step0Output[] run(Configuration conf) throws IOException {

    JobConf job = new JobConf(conf, Step0Job.class);

    // check the output
    if (outputPath.getFileSystem(job).exists(outputPath))
      throw new IOException("Output path already exists : " + outputPath);

    // put the dataset into the DistributedCache
    // use setCacheFiles() to overwrite the first-step cache files
    URI[] files = { datasetPath.toUri() };
    DistributedCache.setCacheFiles(files, job);

    FileInputFormat.setInputPaths(job, dataPath);
    FileOutputFormat.setOutputPath(job, outputPath);

    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(Step0Output.class);

    job.setMapperClass(Step0Mapper.class);
    job.setNumReduceTasks(0); // no reducers

    job.setInputFormat(TextInputFormat.class);
    job.setOutputFormat(SequenceFileOutputFormat.class);

    // run the job
    JobClient.runJob(job);

    return parseOutput(job);
  }

  /**
   * Extracts the output and processes it
   * 
   * @param job
   * 
   * @return firstIds for each partition in Hadoop's order
   * @throws IOException
   */
  protected Step0Output[] parseOutput(JobConf job) throws IOException {
    int numMaps = job.getNumMapTasks();
    FileSystem fs = outputPath.getFileSystem(job);

    Path[] outfiles = DFUtils.listOutputFiles(fs, outputPath);

    int[] keys = new int[numMaps];
    Step0Output[] values = new Step0Output[numMaps];

    // read all the outputs
    IntWritable key = new IntWritable();
    Step0Output value = new Step0Output(0L, 0);

    int index = 0;
    for (Path path : outfiles) {
      Reader reader = new Reader(fs, path, job);

      try {
        while (reader.next(key, value)) {
          keys[index] = key.get();
          values[index] = value.clone();

          index++;
        }
      } finally {
        reader.close();
      }
    }

    return processOutput(keys, values);
  }

  /**
   * Replaces the first id for each partition in Hadoop's order
   * 
   * @param keys
   * @param values
   * @return
   */
  protected static Step0Output[] processOutput(int[] keys, Step0Output[] values) {
    int numMaps = values.length;

    // sort the values using firstId
    Step0Output[] sorted = Arrays.copyOf(values, numMaps);
    Arrays.sort(sorted);

    // compute the partitions firstIds (file order)
    int[] orderedIds = new int[numMaps];
    orderedIds[0] = 0;
    for (int p = 1; p < numMaps; p++) {
      orderedIds[p] = orderedIds[p - 1] + sorted[p - 1].size;
    }

    // update the values' first ids
    for (int p = 0; p < numMaps; p++) {
      int order = ArrayUtils.indexOf(sorted, values[p]);
      values[p].firstId = orderedIds[order];
    }
    
    // reorder the values in hadoop's order
    Step0Output[] reordered = new Step0Output[numMaps];
    for (int p = 0; p < numMaps; p++) {
      reordered[keys[p]] = values[p];
    }

    return reordered;
  }

  /**
   * Outputs the first key and the size of the partition
   * 
   */
  protected static class Step0Mapper extends MapReduceBase implements
      Mapper<LongWritable, Text, IntWritable, Step0Output> {

    protected int partition;

    protected int size;

    protected Long firstId;

    protected OutputCollector<IntWritable, Step0Output> collector;

    @Override
    public void configure(JobConf job) {
      configure(job.getInt("mapred.task.partition", -1));
    }

    /**
     * Useful when testing
     * 
     * @param p
     */
    protected void configure(int p) {
      partition = p;
      if (partition < 0) {
        throw new IllegalArgumentException("Wrong partition id : " + partition);
      }
    }

    @Override
    public void map(LongWritable key, Text value,
        OutputCollector<IntWritable, Step0Output> output, Reporter reporter)
        throws IOException {
      if (firstId == null) {
        firstId = key.get();
      }

      if (collector == null) {
        collector = output;
      }

      size++;
    }

    @Override
    public void close() throws IOException {
      collector.collect(new IntWritable(partition), new Step0Output(firstId,
          size));
    }

  }

  /**
   * Output of the step0's mappers
   * 
   */
  public static class Step0Output implements Writable,
      Comparable<Step0Output> {

    /**
     * first key of the partition<br>
     * used to sort the partition
     */
    protected long firstId;

    /** number of instances in the partition */
    protected int size;

    protected Step0Output(long firstId, int size) {
      this.firstId = firstId;
      this.size = size;
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
    protected Step0Output clone() {
      return new Step0Output(firstId, size);
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof Step0Output)) {
        return false;
      }
      return firstId == ((Step0Output) other).firstId;
    }

    @Override
    public int hashCode() {
      return (int) firstId;
    }

    @Override
    public int compareTo(Step0Output obj) {
      if (firstId < obj.firstId)
        return -1;
      else if (firstId > obj.firstId)
        return 1;
      else
        return 0;
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
