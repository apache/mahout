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
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.regex.Pattern;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
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
  
  private PFPGrowth() { }
  
  /**
   * Generates the fList from the serialized string representation
   *
   * @return Deserialized Feature Frequency List
   */
  public static List<Pair<String,Long>> deserializeList(Parameters params,
                                                        String key,
                                                        Configuration conf) throws IOException {
    List<Pair<String,Long>> list = Lists.newArrayList();
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
                                  + "org.apache.hadoop.io.serializer.WritableSerialization");
    
    DefaultStringifier<List<Pair<String,Long>>> listStringifier = new DefaultStringifier<List<Pair<String,Long>>>(
        conf, GenericsUtil.getClass(list));
    String serializedString = params.get(key, listStringifier.toString(list));
    list = listStringifier.fromString(serializedString);
    return list;
  }
  
  /**
   * Generates the gList(Group ID Mapping of Various frequent Features) Map from the corresponding serialized
   * representation
   *
   * @return Deserialized Group List
   */
  public static Map<String,Long> deserializeMap(Parameters params, String key, Configuration conf) throws IOException {
    Map<String,Long> map = Maps.newHashMap();
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
                                  + "org.apache.hadoop.io.serializer.WritableSerialization");
    
    DefaultStringifier<Map<String,Long>> mapStringifier = new DefaultStringifier<Map<String,Long>>(conf,
        GenericsUtil.getClass(map));
    String gListString = params.get(key, mapStringifier.toString(map));
    map = mapStringifier.fromString(gListString);
    return map;
  }
  
  /**
   * read the feature frequency List which is built at the end of the Parallel counting job
   * 
   * @return Feature Frequency List
   */
  public static List<Pair<String,Long>> readFList(Parameters params) {
    int minSupport = Integer.valueOf(params.get(MIN_SUPPORT, "3"));
    Configuration conf = new Configuration();

    PriorityQueue<Pair<String,Long>> queue =
        new PriorityQueue<Pair<String,Long>>(11, new CountDescendingPairComparator<String,Long>());

    Path parallelCountingPath = new Path(params.get(OUTPUT), PARALLEL_COUNTING);
    Path filesPattern = new Path(parallelCountingPath, FILE_PATTERN);
    for (Pair<Writable,LongWritable> record
         : new SequenceFileDirIterable<Writable,LongWritable>(filesPattern, PathType.GLOB, null, null, true, conf)) {
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
  public static void runPFPGrowth(Parameters params)
    throws IOException, InterruptedException, ClassNotFoundException {
    startParallelCounting(params);
    startGroupingItems(params);
    startTransactionSorting(params);
    startParallelFPGrowth(params);
    startAggregating(params);
  }
  
  /**
   * Run the aggregation Job to aggregate the different TopK patterns and group each Pattern by the features
   * present in it and thus calculate the final Top K frequent Patterns for each feature
   */
  public static void startAggregating(Parameters params)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    Configuration conf = new Configuration();
    params.set(F_LIST, "");
    params.set(G_LIST, "");
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
  public static void startGroupingItems(Parameters params) throws IOException {
    Configuration conf = new Configuration();
    List<Pair<String,Long>> fList = readFList(params);
    Integer numGroups = Integer.valueOf(params.get(NUM_GROUPS, "50"));
    
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
    
    params.set(G_LIST, serializeMap(gList, conf));
    params.set(F_LIST, serializeList(fList, conf));
  }
  
  /**
   * Count the frequencies of various features in parallel using Map/Reduce
   */
  public static void startParallelCounting(Parameters params)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    Configuration conf = new Configuration();
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
  public static void startTransactionSorting(Parameters params)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    Configuration conf = new Configuration();
    String gList = params.get(G_LIST);
    params.set(G_LIST, "");
    conf.set(PFP_PARAMETERS, params.toString());
    conf.set("mapred.compress.map.output", "true");
    conf.set("mapred.output.compression.type", "BLOCK");
    String input = params.get(INPUT);
    Job job = new Job(conf, "PFP Transaction Sorting running over input" + input);
    job.setJarByClass(PFPGrowth.class);
    
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
    params.set(G_LIST, gList);
  }
  
  /**
   * Run the Parallel FPGrowth Map/Reduce Job to calculate the Top K features of group dependent shards
   */
  public static void startParallelFPGrowth(Parameters params)
    throws IOException, InterruptedException, ClassNotFoundException {
    
    Configuration conf = new Configuration();
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
  
  /**
   * Serializes the fList and returns the string representation of the List
   *
   * @return Serialized String representation of List
   */
  private static String serializeList(List<Pair<String,Long>> list, Configuration conf) throws IOException {
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
                                  + "org.apache.hadoop.io.serializer.WritableSerialization");
    DefaultStringifier<List<Pair<String,Long>>> listStringifier = new DefaultStringifier<List<Pair<String,Long>>>(
        conf, GenericsUtil.getClass(list));
    return listStringifier.toString(list);
  }
  
  /**
   * Converts a given Map in to a String using DefaultStringifier of Hadoop
   * 
   * @return Serialized String representation of the GList Map
   */
  private static String serializeMap(Map<String,Long> map, Configuration conf) throws IOException {
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
                                  + "org.apache.hadoop.io.serializer.WritableSerialization");
    DefaultStringifier<Map<String,Long>> mapStringifier = new DefaultStringifier<Map<String,Long>>(conf,
        GenericsUtil.getClass(map));
    return mapStringifier.toString(map);
  }
}
