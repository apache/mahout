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

package org.apache.mahout.classifier.naivebayes.trainer;

import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

public class NaiveBayesInstanceMapper extends Mapper<Text, VectorWritable, IntWritable, VectorWritable> {
  
  private OpenObjectIntHashMap<String> labelMap = new OpenObjectIntHashMap<String>();
  
  @Override
  protected void map(Text key, VectorWritable value, Context context)
      throws IOException, InterruptedException {
    if (!labelMap.containsKey(key.toString())) {
      context.getCounter("NaiveBayes", "Skipped instance: not in label list");
      return;
    }  
    int label = labelMap.get(key.toString());
    context.write(new IntWritable(label), value);
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    try {
      URI[] localFiles = DistributedCache.getCacheFiles(conf);
      if (localFiles == null || localFiles.length < 1) {
        throw new IllegalArgumentException("missing paths from the DistributedCache");
      }
      Path labelMapFile = new Path(localFiles[0].getPath());
      FileSystem fs = labelMapFile.getFileSystem(conf);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, labelMapFile, conf);
      Writable key = new Text();
      IntWritable value = new IntWritable();

      // key is word value is id
      while (reader.next(key, value)) {
        labelMap.put(key.toString(), value.get());
      }
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
}
