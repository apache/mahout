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
import java.io.StringReader;
import java.net.URI;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.lang.mutable.MutableInt;
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
import org.apache.lucene.analysis.Token;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.math.SparseVector;

/**
 * Converts a document in to a SparseVector
 */
public class PartialVectorGenerator extends MapReduceBase implements
    Reducer<Text,Text,Text,SparseVector> {
  private Analyzer analyzer;
  private Map<String,Integer> dictionary = new HashMap<String,Integer>();
  private FileSystem fs; // local filesystem
  private URI[] localFiles; // local filenames from the distributed cache
  
  @Override
  public void reduce(Text key,
                     Iterator<Text> values,
                     OutputCollector<Text,SparseVector> output,
                     Reporter reporter) throws IOException {
    
    if (values.hasNext()) {
      Text value = values.next();
      TokenStream ts =
          analyzer.tokenStream(key.toString(), new StringReader(value
              .toString()));
      
      Map<String,MutableInt> termFrequency = new HashMap<String,MutableInt>();
      
      Token token = new Token();
      int count = 0;
      while ((token = ts.next(token)) != null) {
        String tk = new String(token.termBuffer(), 0, token.termLength());
        if (termFrequency.containsKey(tk) == false) {
          count += tk.length() + 1;
          termFrequency.put(tk, new MutableInt(0));
        }
        termFrequency.get(tk).increment();
      }
      
      SparseVector vector =
          new SparseVector(key.toString(), Integer.MAX_VALUE, termFrequency
              .size());
      
      for (Entry<String,MutableInt> pair : termFrequency.entrySet()) {
        String tk = pair.getKey();
        if (dictionary.containsKey(tk) == false) continue;
        vector.setQuick(dictionary.get(tk).intValue(), pair.getValue()
            .doubleValue());
      }
      
      output.collect(key, vector);
    }
  }
  
  @Override
  public void configure(JobConf job) {
    super.configure(job);
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      Class<?> cl =
          ccl.loadClass(job.get(DictionaryVectorizer.ANALYZER_CLASS,
              StandardAnalyzer.class.getName()));
      analyzer = (Analyzer) cl.newInstance();
      
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
        dictionary.put(key.toString(), Long.valueOf(value.get()).intValue());
        // System.out.println(key.toString() + "=>" + value.get());
      }
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
  
}
