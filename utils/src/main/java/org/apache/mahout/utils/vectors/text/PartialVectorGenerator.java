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

package org.apache.mahout.utils.vectors.text;

import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.StringTokenizer;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VectorWritable;

/**
 * Converts a document in to a sparse vector
 */
public class PartialVectorGenerator extends MapReduceBase implements
    Reducer<Text,Text,Text,VectorWritable> {
  private Analyzer analyzer;
  private final Map<String,int[]> dictionary = new HashMap<String,int[]>();
  private FileSystem fs; // local filesystem
  private URI[] localFiles; // local filenames from the distributed cache
  
  private final VectorWritable vectorWritable = new VectorWritable();
  
  public void reduce(Text key,
                     Iterator<Text> values,
                     OutputCollector<Text,VectorWritable> output,
                     Reporter reporter) throws IOException {
    if (values.hasNext() == false) return;
    Text value = values.next();
    String valueString = value.toString();
    StringTokenizer stream = new StringTokenizer(valueString, " ");
    
    RandomAccessSparseVector vector =
        new RandomAccessSparseVector(key.toString(), Integer.MAX_VALUE,
            valueString.length() / 5); // guess at initial size
    
    while (stream.hasMoreTokens()) {
      String tk = stream.nextToken();
      if (dictionary.containsKey(tk) == false) continue;
      int tokenKey = dictionary.get(tk)[0];
      vector.setQuick(tokenKey, vector.getQuick(tokenKey) + 1);
      
    }
    
    vectorWritable.set(vector);
    output.collect(key, vectorWritable);
    
  }
  
  @Override
  public void configure(JobConf job) {
    super.configure(job);
    try {
      
      localFiles = DistributedCache.getCacheFiles(job);
      if (localFiles == null || localFiles.length < 1) {
        throw new IllegalArgumentException(
            "missing paths from the DistributedCache");
      }
      Path dictionaryFile = new Path(localFiles[0].getPath());
      fs = dictionaryFile.getFileSystem(job);
      SequenceFile.Reader reader =
          new SequenceFile.Reader(fs, dictionaryFile, job);
      Text key = new Text();
      LongWritable value = new LongWritable();
      
      // key is word value is id
      while (reader.next(key, value)) {
        dictionary.put(key.toString(), new int[] {Long.valueOf(value.get())
            .intValue()});
      }
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
  
}
