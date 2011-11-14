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

package org.apache.mahout.fpm.pfpgrowth;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.regex.Pattern;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 * Parallel FP Growth Driver Class. Runs each stage of PFPGrowth as described in the paper
 * http://infolab.stanford.edu/~echang/recsys08-69.pdf
 * 
 */
public final class PFPGrowth {
  
  public static final String ENCODING = "encoding";
  public static final String F_LIST = "fList";
  public static final String G_LIST = "gList";
  public static final String NUM_GROUPS = "numGroups";
  public static final int NUM_GROUPS_DEFAULT = 1000;
  public static final String OUTPUT = "output";
  public static final String MIN_SUPPORT = "minSupport";
  public static final String MAX_HEAPSIZE = "maxHeapSize";
  public static final String INPUT = "input";
  public static final String PFP_PARAMETERS = "pfp.parameters";
  public static final String FILE_PATTERN = "part-*";
  public static final String FPGROWTH = "fpgrowth";
  public static final String FREQUENT_PATTERNS = "frequentpatterns";
  public static final String PARALLEL_COUNTING = "parallelcounting";
  public static final String SORTED_OUTPUT = "sortedoutput";
  public static final String SPLIT_PATTERN = "splitPattern";

  public static final Pattern SPLITTER = Pattern.compile("[ ,\t]*[,|\t][ ,\t]*");
  
  private static final Logger log = LoggerFactory.getLogger(PFPGrowth.class);
  
  private PFPGrowth() {}
  
  /**
   * Generates the fList from the serialized string representation
   * 
   * @return Deserialized Feature Frequency List
   */
  public static List<Pair<String,Long>> readFList(Configuration conf) throws IOException {
    List<Pair<String,Long>> list = new ArrayList<Pair<String,Long>>();
    URI[] files = DistributedCache.getCacheFiles(conf);
    if (files == null || files.length < 2) {
      throw new IOException("Cannot read Frequency list and Grouping list from Distributed Cache");
    }
    for (Pair<Text,LongWritable> record :
         new SequenceFileIterable<Text,LongWritable>(new Path(files[0].getPath()), true, conf)) {
      list.add(new Pair<String,Long>(record.getFirst().toString(), record.getSecond().get()));
    }
    return list;
  }
  
  /**
   * Generates the gList(Group ID Mapping of Various frequent Features) Map from the corresponding serialized
   * representation
   * 
   * @return Deserialized Group List
   */
  public static Map<String,Long> readGList(Configuration conf) throws IOException {
    Map<String,Long> map = new HashMap<String,Long>();
    URI[] files = DistributedCache.getCacheFiles(conf);
    if (files == null || files.length < 2) {
      throw new IOException("Cannot read Frequency list and Grouping list from Distributed Cache");
    }
    for (Pair<Text,LongWritable> record :
         new SequenceFileIterable<Text,LongWritable>(new Path(files[1].getPath()), true, conf)) {
      map.put(record.getFirst().toString(), record.getSecond().get());
    }
    return map;
  }
  
  /**
   * Serializes the fList and returns the string representation of the List
   * 
   * @return Serialized String representation of List
   */
  private static void saveFList(Iterable<Pair<String,Long>> flist, Parameters params, Configuration conf)
    throws IOException {
    Path flistPath = new Path(params.get(OUTPUT), F_LIST);
    FileSystem fs = FileSystem.get(conf);
    flistPath = fs.makeQualified(flistPath);
    HadoopUtil.delete(conf, flistPath);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, flistPath, Text.class, LongWritable.class);
    try {
      for (Pair<String,Long> pair : flist) {
        writer.append(new Text(pair.getFirst()), new LongWritable(pair.getSecond()));
      }
    } finally {
      writer.close();
    }
    DistributedCache.addCacheFile(flistPath.toUri(), conf);
  }
  
  /**
   * Converts a given Map in to a String using DefaultStringifier of Hadoop
   * 
   * @return Serialized String representation of the GList Map
   */
  private static void saveGList(Map<String,Long> glist, Parameters params, Configuration conf) throws IOException {
    Path flistPath = new Path(params.get(OUTPUT), G_LIST);
    FileSystem fs = FileSystem.get(conf);
    flistPath = fs.makeQualified(flistPath);
    HadoopUtil.delete(conf, flistPath);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, flistPath, Text.class, LongWritable.class);
    try {
      for (Entry<String,Long> pair : glist.entrySet()) {
        writer.append(new Text(pair.getKey()), new LongWritable(pair.getValue()));
      }
    } finally {
      writer.close();
    }
    DistributedCache.addCacheFile(flistPath.toUri(), conf);
  }
  
  /**
   * read the feature frequency List which is built at the end of the Parallel counting job
   * 
   * @return Feature Frequency List
   */
  public static List<Pair<String,Long>> readFList(Parameters params) {
    int minSupport = Integer.valueOf(params.get(MIN_SUPPORT, "3"));
    Configuration conf = new Configuration();
    
    Path parallelCountingPath = new Path(params.get(OUTPUT), PARALLEL_COUNTING);

    PriorityQueue<Pair<String,Long>> queue = new PriorityQueue<Pair<String,Long>>(11,
        new Comparator<Pair<String,Long>>() {
          @Override
          public int compare(Pair<String,Long> o1, Pair<String,Long> o2) {
            int ret = o2.getSecond().compareTo(o1.getSecond());
            if (ret != 0) {
              return ret;
            }
            return o1.getFirst().compareTo(o2.getFirst());
          }
        });

    for (Pair<Text,LongWritable> record :
         new SequenceFileDirIterable<Text,LongWritable>(new Path(parallelCountingPath, FILE_PATTERN),
                                                        PathType.GLOB, null, null, true, conf)) {
      long value = record.getSecond().get();
      if (value >= minSupport) {
        queue.add(new Pair<String,Long>(record.getFirst().toString(), value));
      }
    }
    List<Pair<String,Long>> fList = Lists.newArrayList();
    while (!queue.isEmpty()) {
      fList.add(queue.poll());
    }
    return fList;
  }
  
  /**
   * Read the Frequent Patterns generated from Text
   * 
   * @return List of TopK patterns for each string frequent feature
   */
  public static List<Pair<String,TopKStringPatterns>> readFrequentPattern(Parameters params) throws IOException {
    
    Configuration conf = new Configuration();
    
    Path frequentPatternsPath = new Path(params.get(OUTPUT), FREQUENT_PATTERNS);
    FileSystem fs = FileSystem.get(frequentPatternsPath.toUri(), conf);
    FileStatus[] outputFiles = fs.globStatus(new Path(frequentPatternsPath, FILE_PATTERN));
    
    List<Pair<String,TopKStringPatterns>> ret = Lists.newArrayList();
    for (FileStatus fileStatus : outputFiles) {
      ret.addAll(FPGrowth.readFrequentPattern(conf, fileStatus.getPath()));
    }
    return ret;
  }
  
  /**
   * 
   * @param params
   *          params should contain input and output locations as a string value, the additional parameters
   *          include minSupport(3), maxHeapSize(50), numGroups(1000)
   */
  public static void runPFPGrowth(Parameters params) throws IOException,
                                                    InterruptedException,
                                                    ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
                                  + "org.apache.hadoop.io.serializer.WritableSerialization");
    startParallelCounting(params, conf);
    startGroupingItems(params, conf);
    startTransactionSorting(params, conf);
    startParallelFPGrowth(params, conf);
    startAggregating(params, conf);
  }
  
  /**
   * Run the aggregation Job to aggregate the different TopK patterns and group each Pattern by the features
   * present in it and thus calculate the final Top K frequent Patterns for each feature
   */
  public static void startAggregating(Parameters params, Configuration conf)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    conf.set(PFP_PARAMETERS, params.toString());
    conf.set("mapred.compress.map.output", "true");
    conf.set("mapred.output.compression.type", "BLOCK");
    
    Path input = new Path(params.get(OUTPUT), FPGROWTH);
    Job job = new Job(conf, "PFP Aggregator Driver running over input: " + input);
    job.setJarByClass(PFPGrowth.class);
    
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(TopKStringPatterns.class);
    
    FileInputFormat.addInputPath(job, input);
    Path outPath = new Path(params.get(OUTPUT), FREQUENT_PATTERNS);
    FileOutputFormat.setOutputPath(job, outPath);
    
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapperClass(AggregatorMapper.class);
    job.setCombinerClass(AggregatorReducer.class);
    job.setReducerClass(AggregatorReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
    HadoopUtil.delete(conf, outPath);
    job.waitForCompletion(true);
  }
  
  /**
   * Group the given Features into g groups as defined by the numGroups parameter in params
   * 
   * @param params
   * @throws IOException
   */
  public static void startGroupingItems(Parameters params, Configuration conf) throws IOException {
    List<Pair<String,Long>> fList = readFList(params);
    int numGroups = params.getInt(NUM_GROUPS, NUM_GROUPS_DEFAULT);
    
    Map<String,Long> gList = Maps.newHashMap();
    long maxPerGroup = fList.size() / numGroups;
    if (fList.size() != maxPerGroup * numGroups) {
      maxPerGroup++;
    }
    
    long i = 0;
    long groupID = 0;
    for (Pair<String,Long> featureFreq : fList) {
      String feature = featureFreq.getFirst();
      if (i / maxPerGroup == groupID) {
        gList.put(feature, groupID);
      } else {
        groupID++;
        gList.put(feature, groupID);
      }
      i++;
    }
    
    log.info("No of Features: {}", fList.size());
    
    saveFList(fList, params, conf);
    saveGList(gList, params, conf);
  }
  
  /**
   * Count the frequencies of various features in parallel using Map/Reduce
   */
  public static void startParallelCounting(Parameters params, Configuration conf)
    throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(PFP_PARAMETERS, params.toString());
    
    conf.set("mapred.compress.map.output", "true");
    conf.set("mapred.output.compression.type", "BLOCK");
    
    String input = params.get(INPUT);
    Job job = new Job(conf, "Parallel Counting Driver running over input: " + input);
    job.setJarByClass(PFPGrowth.class);
    
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(LongWritable.class);
    
    FileInputFormat.addInputPath(job, new Path(input));
    Path outPath = new Path(params.get(OUTPUT), PARALLEL_COUNTING);
    FileOutputFormat.setOutputPath(job, outPath);
    
    HadoopUtil.delete(conf, outPath);
    
    job.setInputFormatClass(TextInputFormat.class);
    job.setMapperClass(ParallelCountingMapper.class);
    job.setCombinerClass(ParallelCountingReducer.class);
    job.setReducerClass(ParallelCountingReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
    job.waitForCompletion(true);
    
  }
  
  /**
   * Run the Parallel FPGrowth Map/Reduce Job to calculate the Top K features of group dependent shards
   */
  public static void startTransactionSorting(Parameters params, Configuration conf)
    throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(PFP_PARAMETERS, params.toString());
    conf.set("mapred.compress.map.output", "true");
    conf.set("mapred.output.compression.type", "BLOCK");
    String input = params.get(INPUT);
    Job job = new Job(conf, "PFP Transaction Sorting running over input" + input);
    job.setJarByClass(PFPGrowth.class);

    Integer numGroups = Integer.valueOf(params.get(NUM_GROUPS, "-1"));
    int numRed = job.getNumReduceTasks();
    if (numGroups < 0) {
      if (NUM_GROUPS_DEFAULT < numRed) {
        params.set(NUM_GROUPS, Integer.toString(numRed));
      }
    }
    if (numRed > numGroups) {
      log.info("Warning: running with less groups of work than  reducers!");
    }

    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(TransactionTree.class);
    
    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(TransactionTree.class);
    
    FileInputFormat.addInputPath(job, new Path(input));
    Path outPath = new Path(params.get(OUTPUT), SORTED_OUTPUT);
    FileOutputFormat.setOutputPath(job, outPath);
    
    HadoopUtil.delete(conf, outPath);
    
    job.setInputFormatClass(TextInputFormat.class);
    job.setMapperClass(TransactionSortingMapper.class);
    job.setReducerClass(TransactionSortingReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
    job.waitForCompletion(true);
  }
  
  /**
   * Run the Parallel FPGrowth Map/Reduce Job to calculate the Top K features of group dependent shards
   */
  public static void startParallelFPGrowth(Parameters params, Configuration conf)
    throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(PFP_PARAMETERS, params.toString());
    conf.set("mapred.compress.map.output", "true");
    conf.set("mapred.output.compression.type", "BLOCK");
    Path input = new Path(params.get(OUTPUT), SORTED_OUTPUT);
    Job job = new Job(conf, "PFP Growth Driver running over input" + input);
    job.setJarByClass(PFPGrowth.class);
    
    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(TransactionTree.class);
    
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(TopKStringPatterns.class);
    
    FileInputFormat.addInputPath(job, input);
    Path outPath = new Path(params.get(OUTPUT), FPGROWTH);
    FileOutputFormat.setOutputPath(job, outPath);
    
    HadoopUtil.delete(conf, outPath);
    
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapperClass(ParallelFPGrowthMapper.class);
    job.setCombinerClass(ParallelFPGrowthCombiner.class);
    job.setReducerClass(ParallelFPGrowthReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
    job.waitForCompletion(true);
  }
}
