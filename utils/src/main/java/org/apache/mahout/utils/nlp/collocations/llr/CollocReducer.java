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
public class CollocReducer extends MapReduceBase implements Reducer<GramKey,Gram,Gram,Gram> {

  private static final Logger log = LoggerFactory.getLogger(CollocReducer.class);

  public static final String MIN_SUPPORT = "minSupport";
  public static final int DEFAULT_MIN_SUPPORT = 2;
  
  public enum Skipped {
    LESS_THAN_MIN_SUPPORT,
    MALFORMED_KEY_TUPLE,
    MALFORMED_TUPLE,
    MALFORMED_TYPES,
    MALFORMED_UNIGRAM
  }

  private int minSupport;

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    
    this.minSupport = job.getInt(MIN_SUPPORT, DEFAULT_MIN_SUPPORT);

    boolean emitUnigrams = job.getBoolean(CollocDriver.EMIT_UNIGRAMS, CollocDriver.DEFAULT_EMIT_UNIGRAMS);
    
    log.info("Min support is {}", minSupport);
    log.info("Emit Unitgrams is {}", emitUnigrams);

  }
  
  /**
   * collocation finder: pass 1 reduce phase:
   * <p/>
   * given input from the mapper, 
   * 
   * <pre>
   * k:head_subgram,ngram,  v:ngram:partial freq
   * k:head_subgram         v:head_subgram:partial freq
   * k:tail_subgram,ngram,  v:ngram:partial freq
   * k:tail_subgram         v:tail_subgram:partial freq
   * k:unigram              v:unigram:partial freq
   * </pre>
   * sum gram frequencies and output for llr calculation
   * <p/>
   * output is:
   * <pre>
   * k:ngram:ngramfreq      v:head_subgram:head_subgramfreq
   * k:ngram:ngramfreq      v:tail_subgram:tail_subgramfreq
   * k:unigram:unigramfreq  v:unigram:unigramfreq
   * </pre>
   * Each ngram's frequency is essentially counted twice, once for head, once for tail. 
   * frequency should be the same for the head and tail. Fix this to count only for the 
   * head and move the count into the value?
   */
  @Override
  public void reduce(GramKey key,
                     Iterator<Gram> values,
                     OutputCollector<Gram,Gram> output,
                     Reporter reporter) throws IOException {
    
    Gram.Type keyType = key.getType();

    if (keyType == Gram.Type.UNIGRAM) {
      // sum frequencies for unigrams.
      processUnigram(key, values, output, reporter);
    }
    else if (keyType == Gram.Type.HEAD || keyType == Gram.Type.TAIL) {
      // sum frequencies for subgrams, ngram and collect for each ngram.
      processSubgram(key, values, output, reporter);
    }
    else {
      reporter.incrCounter(Skipped.MALFORMED_TYPES, 1);
    }
  }

  /** Sum frequencies for unigrams and deliver to the collector 
   * 
   * @param keyFirst
   * @param values
   * @param output
   * @param reporter
   * @throws IOException
   */
  protected void processUnigram(GramKey key, Iterator<Gram> values,
      OutputCollector<Gram, Gram> output, Reporter reporter) throws IOException {

    int freq = 0;
    Gram value = null;
    
    // accumulate frequencies from values.
    while (values.hasNext()) {
      value = values.next();
      freq += value.getFrequency();
    }

    if (freq < minSupport) {
      reporter.incrCounter(Skipped.LESS_THAN_MIN_SUPPORT, 1);
      return;
    }

    value.setFrequency(freq);
    output.collect(value, value);

  }
      
  /** Sum frequencies for subgram, ngrams and deliver ngram, subgram pairs to the collector.
   *  <p/>
   *  Sort order guarantees that the subgram/subgram pairs will be seen first and then
   *  subgram/ngram1 pairs, subgram/ngram2 pairs ... subgram/ngramN pairs, so frequencies for
   *  ngrams can be calcualted here as well.
   *  <p/>
   *  We end up calculating frequencies for ngrams for each sugram (head, tail) here, which is
   *  some extra work.
   *  
   * @param keyFirst
   * @param values
   * @param output
   * @param reporter
   * @throws IOException
   */
  protected void processSubgram(GramKey key, Iterator<Gram> values, 
      OutputCollector<Gram,Gram> output, Reporter reporter) throws IOException {

    Gram subgram      = null;
    Gram currentNgram = null;
        
    while (values.hasNext()) {
      Gram value = values.next();

      if (value.getType() == Gram.Type.HEAD || value.getType() == Gram.Type.TAIL) { 
        // collect frequency for subgrams.
        if (subgram == null) {
          subgram = new Gram(value);
        }
        else {
          subgram.incrementFrequency(value.getFrequency());
        }
      }
      else if (!value.equals(currentNgram)) {
        // we've collected frequency for all subgrams and we've encountered a new ngram. 
        // collect the old ngram if there was one and we have sufficient support and
        // create the new ngram.
        if (currentNgram != null) {
          if (currentNgram.getFrequency() < minSupport) {
            reporter.incrCounter(Skipped.LESS_THAN_MIN_SUPPORT, 1);
          }
          else {
            output.collect(currentNgram, subgram);
          }
        }

        currentNgram = new Gram(value);
      }
      else {
        currentNgram.incrementFrequency(value.getFrequency());
      }
    }
    
    // collect last ngram.
    if (currentNgram != null) {
      if (currentNgram.getFrequency() < minSupport) {
        reporter.incrCounter(Skipped.LESS_THAN_MIN_SUPPORT, 1);
        return;
      }
      
      output.collect(currentNgram, subgram);
    }
  }
}
