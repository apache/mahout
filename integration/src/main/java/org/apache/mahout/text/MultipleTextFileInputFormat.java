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

package org.apache.mahout.text;

import java.io.IOException;

import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.CombineFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.CombineFileRecordReader;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;

/**
 * 
 * Used in combining a large number of text files into one text input reader
 * along with the WholeFileRecordReader class.
 * 
 */
public class MultipleTextFileInputFormat extends CombineFileInputFormat<IntWritable, BytesWritable> {

  @Override
  public RecordReader<IntWritable, BytesWritable> createRecordReader(InputSplit inputSplit,
                                                                      TaskAttemptContext taskAttemptContext)
      throws IOException {
    return new CombineFileRecordReader<IntWritable, BytesWritable>((CombineFileSplit) inputSplit,
      taskAttemptContext, WholeFileRecordReader.class);
  }
}
