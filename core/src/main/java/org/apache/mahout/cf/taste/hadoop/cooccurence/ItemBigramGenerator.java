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
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.VIntWritable;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ItemBigramGenerator extends Configured implements Tool {
  
  private static final Logger log = LoggerFactory.getLogger(ItemBigramGenerator.class);
  
  public static class UserItemMapper extends MapReduceBase implements
      Mapper<LongWritable,Text,VIntWritable,VIntWritable> {
    
    private static final Logger log = LoggerFactory.getLogger(UserItemMapper.class);
    
    private final VIntWritable user = new VIntWritable(0);
    private final VIntWritable item = new VIntWritable(0);
    
    private String fieldSeparator;
    
    enum Records {
      INVALID_IDS,
      INVALID_SCHEMA
    }
    
    @Override
    public void configure(JobConf conf) {
      fieldSeparator = conf.get("user.preference.field.separator", "\t");
    }
    
    @Override
    public void map(LongWritable lineNumber,
                    Text userPrefEntry,
                    OutputCollector<VIntWritable,VIntWritable> output,
                    Reporter reporter) throws IOException {
      String userPrefLine = userPrefEntry.toString();
      String[] prefFields = userPrefLine.split(fieldSeparator);
      if (prefFields.length > 1) {
        try {
          int userId = Integer.parseInt(prefFields[0]);
          int itemId = Integer.parseInt(prefFields[1]);
          user.set(userId);
          item.set(itemId);
          output.collect(user, item);
        } catch (NumberFormatException nfe) {
          reporter.incrCounter(Records.INVALID_IDS, 1);
          UserItemMapper.log.warn("Invalid IDs in record: {}", userPrefLine);
        } catch (IllegalArgumentException iae) {
          reporter.incrCounter(Records.INVALID_IDS, 1);
          UserItemMapper.log.warn("Invalid IDs in record: {}", userPrefLine);
        }
      } else {
        reporter.incrCounter(Records.INVALID_SCHEMA, 1);
        UserItemMapper.log.warn("No preference found in record: {}", userPrefLine);
      }
    }
  }
  
  public static class UserItemReducer extends MapReduceBase implements
      Reducer<VIntWritable,VIntWritable,VIntWritable,VIntWritable> {
    
    enum User {
      TOO_FEW_ITEMS,
      TOO_MANY_ITEMS
    }
    
    @Override
    public void reduce(VIntWritable user,
                       Iterator<VIntWritable> itemIterator,
                       OutputCollector<VIntWritable,VIntWritable> output,
                       Reporter reporter) throws IOException {
      Collection<VIntWritable> itemSet = new HashSet<VIntWritable>();
      while (itemIterator.hasNext()) {
        itemSet.add(new VIntWritable(itemIterator.next().get()));
      }
      
      if (itemSet.size() <= 2) {
        reporter.incrCounter(User.TOO_FEW_ITEMS, 1);
        return;
      }
      
      if (itemSet.size() >= 10000) {
        reporter.incrCounter(User.TOO_MANY_ITEMS, 1);
        return;
      }
      
      VIntWritable[] items = itemSet.toArray(new VIntWritable[itemSet.size()]);
      
      for (int i = 0; i < items.length; i++) {
        for (int j = i + 1; j < items.length; j++) {
          if (i != j) {
            output.collect(items[i], items[j]);
            output.collect(items[j], items[i]);
          }
        }
      }
    }
    
  }
  
  public JobConf prepareJob(String inputPaths, Path outputPath, int reducers) {
    JobConf job = new JobConf(getConf());
    job.setJarByClass(this.getClass());
    job.setJobName("Item Bigram Generator");
    
    job.setMapperClass(UserItemMapper.class);
    job.setReducerClass(UserItemReducer.class);
    
    job.setOutputKeyClass(VIntWritable.class);
    job.setOutputValueClass(VIntWritable.class);
    
    job.setOutputFormat(SequenceFileOutputFormat.class);
    FileOutputFormat.setCompressOutput(job, true);
    FileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
    SequenceFileOutputFormat.setOutputCompressionType(job, SequenceFile.CompressionType.BLOCK);
    
    job.setNumReduceTasks(reducers);
    
    FileInputFormat.addInputPaths(job, inputPaths);
    FileOutputFormat.setOutputPath(job, outputPath);
    return job;
  }
  
  @Override
  public int run(String[] args) throws IOException {
    // TODO use Commons CLI 2
    if (args.length < 2) {
      ItemBigramGenerator.log.error("Usage: ItemBigramGemerator <input-dir> <output-dir> [reducers]");
      ToolRunner.printGenericCommandUsage(System.out);
      return -1;
    }
    
    String inputPaths = args[0];
    Path outputPath = new Path(args[1]);
    int reducers = args.length > 2 ? Integer.parseInt(args[2]) : 1;
    JobConf jobConf = prepareJob(inputPaths, outputPath, reducers);
    JobClient.runJob(jobConf);
    return 0;
  }
  
}
