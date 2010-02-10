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

package org.apache.mahout.utils.vectors.text.term;

import java.io.IOException;
import java.net.URI;
import java.util.Iterator;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.mahout.common.StringTuple;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenObjectIntHashMap;
import org.apache.mahout.utils.nlp.collocations.llr.CollocMapper.IteratorTokenStream;
import org.apache.mahout.utils.vectors.text.DictionaryVectorizer;

/**
 * Converts a document in to a sparse vector
 */
public class TFPartialVectorReducer extends MapReduceBase implements
    Reducer<Text,StringTuple,Text,VectorWritable> {
  private final OpenObjectIntHashMap<String> dictionary = new OpenObjectIntHashMap<String>();
  
  private final VectorWritable vectorWritable = new VectorWritable();
  
  private int maxNGramSize = 1;
  
  @Override
  public void reduce(Text key,
                     Iterator<StringTuple> values,
                     OutputCollector<Text,VectorWritable> output,
                     Reporter reporter) throws IOException {
    if (values.hasNext() == false) return;
    StringTuple value = values.next();
    
    Vector vector = new RandomAccessSparseVector(key.toString(),
        Integer.MAX_VALUE, value.length()); // guess at initial size
    
    if (maxNGramSize >= 2) {
      ShingleFilter sf = new ShingleFilter(new IteratorTokenStream(value
          .getEntries().iterator()), maxNGramSize);
      
      do {
        String term = ((TermAttribute) sf.getAttribute(TermAttribute.class))
            .term();
        if (term.length() > 0) { // ngram
          if (dictionary.containsKey(term) == false) continue;
          int termId = dictionary.get(term);
          vector.setQuick(termId, vector.getQuick(termId) + 1);
        }
      } while (sf.incrementToken());
      
      sf.end();
      sf.close();
    } else {
      for (String term : value.getEntries()) {
        if (term.length() > 0) { // unigram
          if (dictionary.containsKey(term) == false) continue;
          int termId = dictionary.get(term);
          vector.setQuick(termId, vector.getQuick(termId) + 1);
        }
      }
    }
    vectorWritable.set(vector);
    output.collect(key, vectorWritable);
    
  }
  
  @Override
  public void configure(JobConf job) {
    super.configure(job);
    try {
      maxNGramSize = job.getInt(DictionaryVectorizer.MAX_NGRAMS, maxNGramSize);
      URI[] localFiles = DistributedCache.getCacheFiles(job);
      if (localFiles == null || localFiles.length < 1) {
        throw new IllegalArgumentException(
            "missing paths from the DistributedCache");
      }
      Path dictionaryFile = new Path(localFiles[0].getPath());
      FileSystem fs = dictionaryFile.getFileSystem(job);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, dictionaryFile,
          job);
      Text key = new Text();
      IntWritable value = new IntWritable();
      
      // key is word value is id
      while (reader.next(key, value)) {
        dictionary.put(key.toString(), value.get());
      }
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }
  
}
