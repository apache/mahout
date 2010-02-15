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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.VIntWritable;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.lib.MultipleInputs;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class UserItemJoiner extends Configured implements Tool {
  
  private static final Logger log = LoggerFactory.getLogger(UserItemJoiner.class);
  
  public static class JoinUserMapper extends MapReduceBase implements
      Mapper<LongWritable,Text,Bigram,TupleWritable> {
    
    private static final Logger log = LoggerFactory.getLogger(JoinUserMapper.class);
    
    private final Bigram joinKey = new Bigram();
    private final TupleWritable tuple = new TupleWritable(2);
    private final VIntWritable user = new VIntWritable();
    
    private final VIntWritable keyID = new VIntWritable(1);
    private String fieldSeparator;
    
    @Override
    public void configure(JobConf conf) {
      fieldSeparator = conf.get("user.preference.field.separator", "\t");
    }
    
    @Override
    public void map(LongWritable lineNumber,
                    Text userPrefEntry,
                    OutputCollector<Bigram,TupleWritable> output,
                    Reporter reporter) throws IOException {
      String userPrefLine = userPrefEntry.toString();
      String[] prefFields = userPrefLine.split(fieldSeparator);
      if (prefFields.length > 1) {
        int userId = Integer.parseInt(prefFields[0]);
        int itemId = Integer.parseInt(prefFields[1]);
        user.set(userId);
        tuple.set(0, keyID);
        tuple.set(1, user);
        joinKey.set(itemId, keyID.get());
        output.collect(joinKey, tuple);
      } else {
        log.warn("No preference found in record: {}", userPrefLine);
      }
    }
  }
  
  public static class JoinItemMapper extends MapReduceBase implements
      Mapper<Bigram,DoubleWritable,Bigram,TupleWritable> {
    
    private final VIntWritable simItem = new VIntWritable();
    private final TupleWritable tuple = new TupleWritable(3);
    private final Bigram joinKey = new Bigram();
    
    private final VIntWritable keyID = new VIntWritable(0);
    
    @Override
    public void map(Bigram itemBigram,
                    DoubleWritable simScore,
                    OutputCollector<Bigram,TupleWritable> output,
                    Reporter reporter) throws IOException {
      joinKey.set(itemBigram.getFirst(), keyID.get());
      simItem.set(itemBigram.getSecond());
      tuple.set(0, keyID);
      tuple.set(1, simItem);
      tuple.set(2, simScore);
      output.collect(joinKey, tuple);
    }
  }
  
  public static class JoinItemUserReducer extends MapReduceBase implements
      Reducer<Bigram,TupleWritable,VIntWritable,TupleWritable> {
    
    private final Collection<TupleWritable> cachedSimilarItems = new ArrayList<TupleWritable>();
    
    private final VIntWritable user = new VIntWritable();
    
    @Override
    public void reduce(Bigram itemBigram,
                       Iterator<TupleWritable> tuples,
                       OutputCollector<VIntWritable,TupleWritable> output,
                       Reporter reporter) throws IOException {
      int seenItem = itemBigram.getFirst();
      while (tuples.hasNext()) {
        TupleWritable curTuple = tuples.next();
        int curKeyId = curTuple.getInt(0);
        if (curKeyId == 0) {
          TupleWritable cachedTuple = new TupleWritable(3);
          int simItem = curTuple.getInt(1);
          double score = curTuple.getDouble(2);
          cachedTuple.set(0, new VIntWritable(seenItem));
          cachedTuple.set(1, new VIntWritable(simItem));
          cachedTuple.set(2, new DoubleWritable(score));
          cachedSimilarItems.add(cachedTuple);
        } else {
          // Encountered tuple from the 'user' relation (ID=1), Do the JOIN
          int userId = curTuple.getInt(1);
          user.set(userId);
          for (TupleWritable candItem : cachedSimilarItems) {
            output.collect(user, candItem);
            // System.out.println("User:" + user + "\tSeen:" + candItem.getInt(0) +
            // "\tSimilar:" + candItem.getInt(1) + "\tScore:" + candItem.getDouble(2));
          }
        }
      }
      cachedSimilarItems.clear();
    }
    
  }
  
  public JobConf prepareJob(Path userInputPath, Path itemInputPath, Path outputPath, int reducers) {
    JobConf job = new JobConf(getConf());
    job.setJobName("User Item Joiner");
    job.setJarByClass(this.getClass());
    
    MultipleInputs.addInputPath(job, userInputPath, TextInputFormat.class, JoinUserMapper.class);
    MultipleInputs.addInputPath(job, itemInputPath, SequenceFileInputFormat.class, JoinItemMapper.class);
    job.setReducerClass(JoinItemUserReducer.class);
    
    FileOutputFormat.setOutputPath(job, outputPath);
    
    job.setMapOutputKeyClass(Bigram.class);
    job.setMapOutputValueClass(TupleWritable.class);
    job.setOutputKeyClass(VIntWritable.class);
    job.setOutputValueClass(TupleWritable.class);
    
    job.setOutputFormat(SequenceFileOutputFormat.class);
    FileOutputFormat.setCompressOutput(job, true);
    FileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
    SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);
    
    job.setPartitionerClass(ItemSimilarityEstimator.FirstPartitioner.class);
    job.setOutputValueGroupingComparator(Bigram.FirstGroupingComparator.class);
    
    job.setNumReduceTasks(reducers);
    
    return job;
  }
  
  @Override
  public int run(String[] args) throws IOException {
    // TODO use Commons CLI 2
    if (args.length < 3) {
      log.error("UserItemJoiner <user-input-dirs> <item-input-dir> <output-dir> [reducers]");
      ToolRunner.printGenericCommandUsage(System.out);
      return -1;
    }
    
    Path userInputPath = new Path(args[0]);
    Path itemInputPath = new Path(args[1]);
    Path outputPath = new Path(args[2]);
    int reducers = args.length > 3 ? Integer.parseInt(args[3]) : 1;
    JobConf jobConf = prepareJob(userInputPath, itemInputPath, outputPath, reducers);
    JobClient.runJob(jobConf);
    return 0;
  }
  
}
