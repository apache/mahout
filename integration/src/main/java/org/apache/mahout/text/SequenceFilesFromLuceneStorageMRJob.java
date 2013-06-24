package org.apache.mahout.text;
/*
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

import com.google.common.base.Joiner;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import java.io.IOException;

/**
 * Generates a sequence file from a Lucene index via MapReduce. Uses a specified id field as the key and a content field
 * as the value. Configure this class with a {@link LuceneStorageConfiguration} bean.
 */
public class SequenceFilesFromLuceneStorageMRJob {

  public void run(LuceneStorageConfiguration lucene2seqConf) {
    try {
      Configuration configuration = lucene2seqConf.serialize();

      Job job = new Job(configuration, "LuceneIndexToSequenceFiles: " + lucene2seqConf.getIndexPaths() + " -> M/R -> "
          + lucene2seqConf.getSequenceFilesOutputPath());

      job.setMapOutputKeyClass(Text.class);
      job.setMapOutputValueClass(Text.class);

      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(Text.class);

      job.setOutputFormatClass(SequenceFileOutputFormat.class);

      job.setMapperClass(SequenceFilesFromLuceneStorageMapper.class);

      job.setInputFormatClass(LuceneSegmentInputFormat.class);

      FileInputFormat.setInputPaths(job, Joiner.on(',').skipNulls().join(lucene2seqConf.getIndexPaths().iterator()));
      FileOutputFormat.setOutputPath(job, lucene2seqConf.getSequenceFilesOutputPath());

      job.setJarByClass(SequenceFilesFromLuceneStorageMRJob.class);
      job.setNumReduceTasks(0);

      job.waitForCompletion(true);
    } catch (IOException e) {
      throw new RuntimeException(e);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
}
