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

package org.apache.mahout.utils.nlp.collocations.llr;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reducer for Pass 1 of the collocation identification job. Generates counts for ngrams and subgrams.
 */
public class CollocReducer extends MapReduceBase implements Reducer<Gram,Gram,Gram,Gram> {
  
  public static final String MIN_SUPPORT = "minSupport";
  public static final int DEFAULT_MIN_SUPPORT = 2;
  
  public enum Skipped {
    LESS_THAN_MIN_SUPPORT
  }

  private static final Logger log = LoggerFactory.getLogger(CollocReducer.class);
  
  private int minSupport;
  private boolean emitUnigrams;
  
  @Override
  public void configure(JobConf job) {
    super.configure(job);
    
    this.minSupport = job.getInt(MIN_SUPPORT, DEFAULT_MIN_SUPPORT);
    
    this.emitUnigrams = job.getBoolean(CollocDriver.EMIT_UNIGRAMS, CollocDriver.DEFAULT_EMIT_UNIGRAMS);
    
    if (log.isInfoEnabled()) {
      log.info("Min support is {}", minSupport);
      log.info("Emit Unitgrams is {}", emitUnigrams);
    }
    
  }
  
  /**
   * collocation finder: pass 1 reduce phase:
   * 
   * given input from the mapper, k:h_subgram v:ngram k:t_subgram v:ngram
   * 
   * count ngrams and subgrams.
   * 
   * output is:
   * 
   * k:ngram:ngramfreq v:h_subgram:h_subgramfreq k:ngram:ngramfreq v:t_subgram:t_subgramfreq
   * 
   * Each ngram's frequency is essentially counted twice, frequency should be the same for the head and tail.
   * Fix this to count only for the head and move the count into the value?
   */
  @Override
  public void reduce(Gram subgramKey,
                     Iterator<Gram> ngramValues,
                     OutputCollector<Gram,Gram> output,
                     Reporter reporter) throws IOException {
    
    HashMap<Gram,Gram> ngramSet = new HashMap<Gram,Gram>();
    int subgramFrequency = 0;
    
    while (ngramValues.hasNext()) {
      Gram ngram = ngramValues.next();
      subgramFrequency += ngram.getFrequency();
      
      Gram ngramCanon = ngramSet.get(ngram);
      if (ngramCanon == null) {
        // t is potentially reused, so create a new object to populate the
        // HashMap
        Gram ngramEntry = new Gram(ngram);
        ngramSet.put(ngramEntry, ngramEntry);
      } else {
        ngramCanon.incrementFrequency(ngram.getFrequency());
      }
    }
    
    // emit ngram:ngramFreq, subgram:subgramFreq pairs.
    subgramKey.setFrequency(subgramFrequency);
    
    for (Gram ngram : ngramSet.keySet()) {
      if (ngram.getFrequency() < minSupport) {
        reporter.incrCounter(Skipped.LESS_THAN_MIN_SUPPORT, 1);
        continue;
      }
      output.collect(ngram, subgramKey);
    }
  }
}
