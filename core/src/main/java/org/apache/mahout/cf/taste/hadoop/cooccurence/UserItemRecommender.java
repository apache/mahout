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
package org.apache.mahout.cf.taste.hadoop.cooccurence;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.VIntWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

public final class UserItemRecommender extends Configured implements Tool {

  private static final Logger log = LoggerFactory.getLogger(UserItemRecommender.class);

  public static class RecommenderMapper extends MapReduceBase
      implements Mapper<VIntWritable, TupleWritable, Bigram, TupleWritable> {

    private final Bigram userSeenItem = new Bigram();

    @Override
    public void map(VIntWritable user, TupleWritable tuple, OutputCollector<Bigram, TupleWritable> output,
                    Reporter reporter) throws IOException {
      int userId = user.get();
      int seenItemId = tuple.getInt(0);
      userSeenItem.set(userId, seenItemId);
      output.collect(userSeenItem, tuple);
    }
  }

  public static class RecommenderReducer extends MapReduceBase implements Reducer<Bigram, TupleWritable, Text, Text> {

    private String fieldSeparator;
    private int maxRecommendations;

    private final List<Integer> seenItems = new ArrayList<Integer>();
    private final Map<Integer, Double> recommendations = new HashMap<Integer, Double>();

    private final Text user = new Text();
    private final Text recomScore = new Text();

    private static class EntryValueComparator implements Comparator<Map.Entry<Integer, Double>>, Serializable {

      @Override
      public int compare(Map.Entry<Integer, Double> itemScore1, Map.Entry<Integer, Double> itemScore2) {
        Double value1 = itemScore1.getValue();
        double val1 = (value1 == null) ? 0 : value1;
        Double value2 = itemScore2.getValue();
        double val2 = (value2 == null) ? 0 : value2;
        return val2 > val1 ? 1 : -1;
      }
    }

    @Override
    public void configure(JobConf conf) {
      fieldSeparator = conf.get("user.preference.field.separator", "\t");
      maxRecommendations = conf.getInt("user.preference.max.recommendations", 100);
    }

    @Override
    public void reduce(Bigram userSeenItem, Iterator<TupleWritable> candTupleItr, OutputCollector<Text, Text> output,
                       Reporter reporter)
        throws IOException {
      int userId = userSeenItem.getFirst();
      int prevSeenItem = userSeenItem.getSecond();

      while (candTupleItr.hasNext()) {
        TupleWritable tuple = candTupleItr.next();
        int curSeenItem = tuple.getInt(0);
        if (curSeenItem != prevSeenItem) {
          seenItems.add(prevSeenItem);
          recommendations.remove(prevSeenItem);
          prevSeenItem = curSeenItem;
        }
        int candItem = tuple.getInt(1);
        double score = tuple.getDouble(2);
        if (Collections.binarySearch(seenItems, candItem) < 0) {
          score = recommendations.containsKey(candItem) ? score + recommendations.get(candItem) : score;
          recommendations.put(candItem, score);
        } else {
          recommendations.remove(candItem);
        }
      }
      recommendations.remove(prevSeenItem);
      //Sort recommendations by count and output top-N
      outputSorted(userId, recommendations.entrySet(), output);
    }

    public void outputSorted(int userId, Collection<Map.Entry<Integer, Double>> recomSet, OutputCollector<Text, Text> output)
        throws IOException {
      user.set(String.valueOf(userId));
      int N = maxRecommendations;
      Collection<Map.Entry<Integer, Double>> sortedRecoms =
          new TreeSet<Map.Entry<Integer, Double>>(new EntryValueComparator());
      sortedRecoms.addAll(recomSet);
      for (Map.Entry<Integer, Double> recommendation : sortedRecoms) {
        recomScore.set(recommendation.getKey() + fieldSeparator + recommendation.getValue());
        output.collect(user, recomScore);
        N--;
        if (N <= 0) {
          break;
        }
      }
      seenItems.clear();
      recommendations.clear();
    }
  }

  public JobConf prepareJob(String inputPaths, Path outputPath, int maxRecommendations, int reducers) {
    JobConf job = new JobConf(getConf());
    job.setJobName("User Item Recommendations");
    job.setJarByClass(this.getClass());

    job.setInputFormat(SequenceFileInputFormat.class);
    job.setOutputFormat(TextOutputFormat.class);
    job.setMapperClass(RecommenderMapper.class);
    job.setReducerClass(RecommenderReducer.class);

    job.setMapOutputKeyClass(Bigram.class);
    job.setMapOutputValueClass(TupleWritable.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);

    job.setPartitionerClass(ItemSimilarityEstimator.FirstPartitioner.class);
    job.setOutputValueGroupingComparator(Bigram.FirstGroupingComparator.class);

    job.setInt("user.preference.max.recommendations", maxRecommendations);
    job.setNumReduceTasks(reducers);
    FileInputFormat.addInputPaths(job, inputPaths);
    FileOutputFormat.setOutputPath(job, outputPath);
    return job;
  }

  @Override
  public int run(String[] args) throws IOException {
    // TODO use Commons CLI 2
    if (args.length < 2) {
      log.error("UserItemRecommender <input-dirs> <output-dir> [max-recommendations] [reducers]");
      ToolRunner.printGenericCommandUsage(System.out);
      return -1;
    }

    String inputPaths = args[0];
    Path outputPath = new Path(args[1]);
    int maxRecommendations = args.length > 2 ? Integer.parseInt(args[2]) : 100;
    int reducers = args.length > 3 ? Integer.parseInt(args[3]) : 1;
    JobConf jobConf = prepareJob(inputPaths, outputPath, maxRecommendations, reducers);
    JobClient.runJob(jobConf);
    return 0;
  }

}
