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
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.mahout.common.IntegerTuple;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 
 * Parallel FP Growth Driver Class. Runs each stage of PFPGrowth as described in
 * the paper http://infolab.stanford.edu/~echang/recsys08-69.pdf
 * 
 */
public class PFPGrowth {
  public static final Pattern SPLITTER = Pattern.compile("[ ,\t]*[,|\t][ ,\t]*");

  private static final Logger log = LoggerFactory.getLogger(PFPGrowth.class);

  private PFPGrowth() {
  }

  /**
   * Generates the fList from the serialized string representation
   * 
   * @param params
   * @param key
   * @param conf
   * @return Deserialized Feature Frequency List
   * @throws IOException
   */
  public static List<Pair<String, Long>> deserializeList(Parameters params,
      String key, Configuration conf) throws IOException {
    List<Pair<String, Long>> list = new ArrayList<Pair<String, Long>>();
    conf.set(
            "io.serializations",
            "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");

    DefaultStringifier<List<Pair<String, Long>>> listStringifier = new DefaultStringifier<List<Pair<String, Long>>>(
        conf, GenericsUtil.getClass(list));
    String serializedString = listStringifier.toString(list);
    serializedString = params.get(key, serializedString);
    list = listStringifier.fromString(serializedString);
    return list;
  }

  /**
   * Generates the gList(Group ID Mapping of Various frequent Features) Map from
   * the corresponding serialized representation
   * 
   * @param params
   * @param key
   * @param conf
   * @return Deserialized Group List
   * @throws IOException
   */
  public static Map<String, Long> deserializeMap(Parameters params, String key,
      Configuration conf) throws IOException {
    Map<String, Long> map = new HashMap<String, Long>();
    conf
        .set(
            "io.serializations",
            "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");

    DefaultStringifier<Map<String, Long>> mapStringifier = new DefaultStringifier<Map<String, Long>>(
        conf, GenericsUtil.getClass(map));
    String gListString = mapStringifier.toString(map);
    gListString = params.get(key, gListString);
    map = mapStringifier.fromString(gListString);
    return map;
  }

  /**
   * read the feature frequency List which is built at the end of the Parallel
   * counting job
   * 
   * @param params
   * @return Feature Frequency List
   * @throws IOException
   */
  public static List<Pair<String, Long>> readFList(Parameters params)
      throws IOException {
    Writable key = new Text();
    LongWritable value = new LongWritable();
    int minSupport = Integer.valueOf(params.get("minSupport", "3"));
    Configuration conf = new Configuration();

    FileSystem fs = FileSystem.get(new Path(params.get("output")
        + "/parallelcounting").toUri(), conf);
    FileStatus[] outputFiles = fs.globStatus(new Path(params.get("output")
        + "/parallelcounting/part-*"));

    PriorityQueue<Pair<String, Long>> queue = new PriorityQueue<Pair<String, Long>>(
        11, new Comparator<Pair<String, Long>>() {

          @Override
          public int compare(Pair<String, Long> o1, Pair<String, Long> o2) {
            int ret = o2.getSecond().compareTo(o1.getSecond());
            if (ret != 0)
              return ret;
            return o1.getFirst().compareTo(o2.getFirst());
          }

        });
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      // key is feature value is count
      while (reader.next(key, value)) {
        if (value.get() < minSupport)
          continue;
        queue.add(new Pair<String, Long>(key.toString(), value.get()));
      }
    }
    List<Pair<String, Long>> fList = new ArrayList<Pair<String, Long>>();
    while (queue.isEmpty() == false)
      fList.add(queue.poll());
    return fList;
  }

  /**
   * Read the Frequent Patterns generated from Text
   * 
   * @param params
   * @return List of TopK patterns for each string frequent feature
   * @throws IOException
   */
  public static List<Pair<String, TopKStringPatterns>> readFrequentPattern(
      Parameters params) throws IOException {

    Configuration conf = new Configuration();

    FileSystem fs = FileSystem.get(new Path(params.get("output")
        + "/frequentPatterns").toUri(), conf);
    FileStatus[] outputFiles = fs.globStatus(new Path(params.get("output")
        + "/frequentPatterns/part-*"));

    List<Pair<String, TopKStringPatterns>> ret = new ArrayList<Pair<String, TopKStringPatterns>>();
    for (FileStatus fileStatus : outputFiles) {
      Path path = fileStatus.getPath();
      ret.addAll(FPGrowth.readFrequentPattern(fs, conf, path));
    }
    return ret;
  }

  /**
   * 
   * @param params params should contain input and output locations as a string
   *        value, the additional parameters include minSupport(3),
   *        maxHeapSize(50), numGroups(1000)
   * @throws IOException
   * @throws ClassNotFoundException
   * @throws InterruptedException
   */
  public static void runPFPGrowth(Parameters params) throws IOException,
      InterruptedException, ClassNotFoundException {
    startParallelCounting(params);
    startGroupingItems(params);
    startParallelFPGrowth(params);
    startAggregating(params);
  }

  /**
   * Run the aggregation Job to aggregate the different TopK patterns and group
   * each Pattern by the features present in it and thus calculate the final Top
   * K frequent Patterns for each feature
   * 
   * @param params
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static void startAggregating(Parameters params) throws IOException,
      InterruptedException, ClassNotFoundException {

    Configuration conf = new Configuration();
    params.set("fList", "");
    params.set("gList", "");
    conf.set("pfp.parameters", params.toString());

    String input = params.get("output") + "/fpgrowth";
    Job job = new Job(conf, "PFP Aggregator Driver running over input: "
        + input);
    job.setJarByClass(PFPGrowth.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(TopKStringPatterns.class);

    FileInputFormat.addInputPath(job, new Path(input));
    Path outPath = new Path(params.get("output") + "/frequentPatterns");
    FileOutputFormat.setOutputPath(job, outPath);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapperClass(AggregatorMapper.class);
    job.setCombinerClass(AggregatorReducer.class);
    job.setReducerClass(AggregatorReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }
    job.waitForCompletion(true);
  }

  /**
   * Group the given Features into g groups as defined by the numGroups
   * parameter in params
   * 
   * @param params
   * @throws IOException
   */
  public static void startGroupingItems(Parameters params) throws IOException {
    Configuration conf = new Configuration();
    List<Pair<String, Long>> fList = readFList(params);
    Integer numGroups = Integer.valueOf(params.get("numGroups", "50"));

    Map<String, Long> gList = new HashMap<String, Long>();
    long maxPerGroup = fList.size() / numGroups;
    if (fList.size() != maxPerGroup * numGroups)
      maxPerGroup++;

    long i = 0;
    long groupID = 0;
    for (Pair<String, Long> featureFreq : fList) {
      String feature = featureFreq.getFirst();
      if (i / (maxPerGroup) == groupID) {
        gList.put(feature, groupID);
      } else {
        groupID++;
        gList.put(feature, groupID);
      }
      i++;
    }

    log.info("No of Features: {}", fList.size());

    params.set("gList", serializeMap(gList, conf));
    params.set("fList", serializeList(fList, conf));
  }

  /**
   * Count the frequencies of various features in parallel using Map/Reduce
   * 
   * @param params
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static void startParallelCounting(Parameters params)
      throws IOException, InterruptedException, ClassNotFoundException {

    Configuration conf = new Configuration();
    conf.set("pfp.parameters", params.toString());

    String input = params.get("input");
    Job job = new Job(conf, "Parallel Counting Driver running over input: "
        + input);
    job.setJarByClass(PFPGrowth.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(LongWritable.class);

    FileInputFormat.addInputPath(job, new Path(input));
    Path outPath = new Path(params.get("output") + "/parallelcounting");
    FileOutputFormat.setOutputPath(job, outPath);

    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }

    job.setInputFormatClass(TextInputFormat.class);
    job.setMapperClass(ParallelCountingMapper.class);
    job.setCombinerClass(ParallelCountingReducer.class);
    job.setReducerClass(ParallelCountingReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    job.waitForCompletion(true);

  }

  /**
   * Run the Parallel FPGrowth Map/Reduce Job to calculate the Top K features of
   * group dependent shards
   * 
   * @param params
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public static void startParallelFPGrowth(Parameters params)
      throws IOException, InterruptedException, ClassNotFoundException {

    Configuration conf = new Configuration();
    conf.set("pfp.parameters", params.toString());

    String input = params.get("input");
    Job job = new Job(conf, "PFP Growth Driver running over input" + input);
    job.setJarByClass(PFPGrowth.class);

    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(IntegerTuple.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(TopKStringPatterns.class);

    FileInputFormat.addInputPath(job, new Path(input));
    Path outPath = new Path(params.get("output") + "/fpgrowth");
    FileOutputFormat.setOutputPath(job, outPath);

    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }

    job.setInputFormatClass(TextInputFormat.class);
    job.setMapperClass(ParallelFPGrowthMapper.class);
    job.setReducerClass(ParallelFPGrowthReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    job.waitForCompletion(true);
  }

  /**
   * Serializes the fList and returns the string representation of the List
   * 
   * @param list
   * @param conf
   * @return Serialized String representation of List
   * @throws IOException
   */
  private static String serializeList(List<Pair<String, Long>> list,
      Configuration conf) throws IOException {
    conf.set(
            "io.serializations",
            "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
    DefaultStringifier<List<Pair<String, Long>>> listStringifier = new DefaultStringifier<List<Pair<String, Long>>>(
        conf, GenericsUtil.getClass(list));
    return listStringifier.toString(list);
  }

  /**
   * Converts a given Map in to a String using DefaultStringifier of Hadoop
   * 
   * @param map
   * @param conf
   * @return Serialized String representation of the GList Map
   * @throws IOException
   */
  private static String serializeMap(Map<String, Long> map, Configuration conf)
      throws IOException {
    conf.set(
            "io.serializations",
            "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
    DefaultStringifier<Map<String, Long>> mapStringifier = new DefaultStringifier<Map<String, Long>>(
        conf, GenericsUtil.getClass(map));
    String serializedMapString = mapStringifier.toString(map);
    return serializedMapString;
  }
}
