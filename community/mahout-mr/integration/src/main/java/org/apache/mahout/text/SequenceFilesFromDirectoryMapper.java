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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.CombineFileSplit;
import org.apache.mahout.common.HadoopUtil;

import static org.apache.mahout.text.SequenceFilesFromDirectory.KEY_PREFIX_OPTION;

/**
 * Map class for SequenceFilesFromDirectory MR job
 */
public class SequenceFilesFromDirectoryMapper extends Mapper<IntWritable, BytesWritable, Text, Text> {

  private String keyPrefix;
  private Text fileValue = new Text();

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    this.keyPrefix = context.getConfiguration().get(KEY_PREFIX_OPTION[0], "");
  }

  public void map(IntWritable key, BytesWritable value, Context context)
    throws IOException, InterruptedException {

    Configuration configuration = context.getConfiguration();
    Path filePath = ((CombineFileSplit) context.getInputSplit()).getPath(key.get());
    String relativeFilePath = HadoopUtil.calcRelativeFilePath(configuration, filePath);

    String filename = this.keyPrefix.length() > 0 ?
      this.keyPrefix + Path.SEPARATOR + relativeFilePath :
      Path.SEPARATOR + relativeFilePath;

    fileValue.set(value.getBytes(), 0, value.getBytes().length);
    context.write(new Text(filename), fileValue);
  }
}
