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

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;

/**
 * {@link InputFormat} implementation which splits a Lucene index at the segment level.
 */
public class LuceneSegmentInputFormat extends InputFormat {

  private static final Logger LOG = LoggerFactory.getLogger(LuceneSegmentInputFormat.class);

  @Override
  public List<LuceneSegmentInputSplit> getSplits(JobContext context) throws IOException, InterruptedException {
    Configuration configuration = context.getConfiguration();

    LuceneStorageConfiguration lucene2SeqConfiguration = new LuceneStorageConfiguration(configuration);

    List<LuceneSegmentInputSplit> inputSplits = Lists.newArrayList();

    List<Path> indexPaths = lucene2SeqConfiguration.getIndexPaths();
    for (Path indexPath : indexPaths) {
      ReadOnlyFileSystemDirectory directory = new ReadOnlyFileSystemDirectory(FileSystem.get(configuration), indexPath,
                                                                              false, configuration);
      SegmentInfos segmentInfos = new SegmentInfos();
      segmentInfos.read(directory);

      for (SegmentCommitInfo segmentInfo : segmentInfos) {
        LuceneSegmentInputSplit inputSplit = new LuceneSegmentInputSplit(indexPath, segmentInfo.info.name,
                                                                         segmentInfo.sizeInBytes());
        inputSplits.add(inputSplit);
        LOG.info("Created {} byte input split for index '{}' segment {}", segmentInfo.sizeInBytes(), indexPath.toUri(),
                 segmentInfo.info.name);
      }
    }

    return inputSplits;
  }

  @Override
  public RecordReader<Text, NullWritable> createRecordReader(InputSplit inputSplit, TaskAttemptContext context)
    throws IOException, InterruptedException {
    LuceneSegmentRecordReader luceneSegmentRecordReader = new LuceneSegmentRecordReader();
    luceneSegmentRecordReader.initialize(inputSplit, context);
    return luceneSegmentRecordReader;
  }
}
