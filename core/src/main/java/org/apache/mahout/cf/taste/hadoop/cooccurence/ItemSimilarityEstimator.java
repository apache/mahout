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
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.VIntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Iterator;
import java.util.PriorityQueue;
import java.util.Queue;


/**
 * This class feeds into all the item bigrams generated with ItemBigramGenerator. The input is partitioned on the first
 * item of the bigram, distributed and sorted by the map-reduce framework and grouped on first item of the bigram so
 * that each reducer sees all the bigrams for each unique first item.
 */
public final class ItemSimilarityEstimator extends Configured implements Tool {

  private static final Logger log = LoggerFactory.getLogger(ItemSimilarityEstimator.class);    

  /** Partition based on the first part of the bigram. */
  public static class FirstPartitioner implements Partitioner<Bigram, Writable> {

    @Override
    public int getPartition(Bigram key, Writable value,
                            int numPartitions) {
      return Math.abs(key.getFirst() % numPartitions);
    }

    @Override
    public void configure(JobConf jobConf) {
      // Nothing to do here      
    }
  }

  /** Output K -> (item1, item2), V -> ONE */
  public static class ItemItemMapper extends MapReduceBase
      implements Mapper<VIntWritable, VIntWritable, Bigram, Bigram> {

    private final Bigram keyBigram = new Bigram();
    private final Bigram valueBigram = new Bigram();
    private static final int ONE = 1;

    @Override
    public void map(VIntWritable item1, VIntWritable item2, OutputCollector<Bigram, Bigram> output, Reporter reporter)
        throws IOException {
      keyBigram.set(item1.get(), item2.get());
      valueBigram.set(item2.get(), ONE);
      output.collect(keyBigram, valueBigram);
    }
  }


  /* Test waters */

  public static class ItemItemCombiner extends MapReduceBase implements Reducer<Bigram, Bigram, Bigram, Bigram> {

    @Override
    public void reduce(Bigram item, Iterator<Bigram> similarItemItr,
                       OutputCollector<Bigram, Bigram> output, Reporter reporter)
        throws IOException {
      int count = 0;
      while (similarItemItr.hasNext()) {
        Bigram candItem = similarItemItr.next();
        count += candItem.getSecond();
      }
      Bigram similarItem = new Bigram(item.getSecond(), count);
      output.collect(item, similarItem);
    }
  }

  /** All sorted bigrams for item1 are recieved in reduce. <p/> K -> (item1, item2), V -> (FREQ) */
  public static class ItemItemReducer extends MapReduceBase implements Reducer<Bigram, Bigram, Bigram, DoubleWritable> {

    private final Queue<Bigram.Frequency> freqBigrams = new PriorityQueue<Bigram.Frequency>();
    private Bigram key = new Bigram();
    private DoubleWritable value = new DoubleWritable();

    private long maxFrequentItems;

    @Override
    public void configure(JobConf conf) {
      maxFrequentItems = conf.
          getLong("max.frequent.items", 20);
    }

    @Override
    public void reduce(Bigram item, Iterator<Bigram> simItemItr,
                       OutputCollector<Bigram, DoubleWritable> output, Reporter reporter)
        throws IOException {
      int itemId = item.getFirst();
      int prevItemId = item.getSecond();
      int prevCount = 0;
      while (simItemItr.hasNext()) {
        Bigram curItem = simItemItr.next();
        int curItemId = curItem.getFirst();
        int curCount = curItem.getSecond();
        if (prevItemId == curItemId) {
          prevCount += curCount;
        } else {
          enqueue(itemId, prevItemId, prevCount);
          prevItemId = curItemId;
          prevCount = curCount;
        }
      }
      enqueue(itemId, prevItemId, prevCount);
      dequeueAll(output);
    }

    private void enqueue(int first, int second, int count) {
      Bigram freqBigram = new Bigram(first, second);
      freqBigrams.add(new Bigram.Frequency(freqBigram, count));
      if (freqBigrams.size() > maxFrequentItems) {
        freqBigrams.poll();
      }
    }

    private void dequeueAll(OutputCollector<Bigram, DoubleWritable> output) throws IOException {
      double totalScore = 0;
      for (Bigram.Frequency freqBigram : freqBigrams) {
        totalScore += freqBigram.getFrequency();
      }
      // normalize the co-occurrence based counts.
      for (Bigram.Frequency freqBigram : freqBigrams) {
        key = freqBigram.getBigram();
        value = new DoubleWritable(freqBigram.getFrequency() / totalScore);
        output.collect(key, value);
      }
      freqBigrams.clear();
    }
  }


  public JobConf prepareJob(String inputPaths, Path outputPath, int maxFreqItems, int reducers) {
    JobConf job = new JobConf(getConf());
    job.setJobName("Item Bigram Counter");
    job.setJarByClass(this.getClass());

    job.setMapperClass(ItemItemMapper.class);
    job.setCombinerClass(ItemItemCombiner.class);
    job.setReducerClass(ItemItemReducer.class);

    job.setMapOutputKeyClass(Bigram.class);
    job.setMapOutputValueClass(Bigram.class);
    job.setOutputKeyClass(Bigram.class);
    job.setOutputValueClass(DoubleWritable.class);

    job.setInputFormat(SequenceFileInputFormat.class);
    job.setOutputFormat(SequenceFileOutputFormat.class);
    FileOutputFormat.setCompressOutput(job, true);
    FileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
    SequenceFileOutputFormat.setOutputCompressionType(job,
                                                      SequenceFile.CompressionType.BLOCK);

    job.setPartitionerClass(FirstPartitioner.class);
    job.setOutputValueGroupingComparator(Bigram.FirstGroupingComparator.class);

    job.setInt("max.frequent.items", maxFreqItems);
    job.setNumReduceTasks(reducers);
    FileInputFormat.addInputPaths(job, inputPaths);
    FileOutputFormat.setOutputPath(job, outputPath);
    return job;
  }

  @Override
  public int run(String[] args) throws IOException {
    // TODO use Commons CLI 2
    if (args.length < 2) {
      log.error("ItemSimilarityEstimator <input-dirs> <output-dir> [max-frequent-items] [reducers]");
      ToolRunner.printGenericCommandUsage(System.out);
      return -1;
    }

    String inputPaths = args[0];
    Path outputPath = new Path(args[1]);
    int maxFreqItems = args.length > 2 ? Integer.parseInt(args[2]) : 20;
    int reducers = args.length > 3 ? Integer.parseInt(args[3]) : 1;
    JobConf jobConf = prepareJob(inputPaths, outputPath, maxFreqItems, reducers);
    JobClient.runJob(jobConf);
    return 0;
  }

}
