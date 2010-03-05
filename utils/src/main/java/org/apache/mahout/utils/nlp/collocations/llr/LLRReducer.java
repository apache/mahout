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
import java.util.Iterator;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.stats.LogLikelihood;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reducer for pass 2 of the collocation discovery job. Collects ngram and sub-ngram frequencies and performs
 * the Log-likelihood ratio calculation.
 */
public class LLRReducer extends MapReduceBase implements Reducer<Gram,Gram,Text,DoubleWritable> {
  
  /** Counter to track why a particlar entry was skipped */
  public enum Skipped {
    EXTRA_HEAD,
    EXTRA_TAIL,
    MISSING_HEAD,
    MISSING_TAIL,
    LESS_THAN_MIN_LLR,
    LLR_CALCULATION_ERROR,
  }

  private static final Logger log = LoggerFactory.getLogger(LLRReducer.class);
  
  public static final String NGRAM_TOTAL = "ngramTotal";
  public static final String MIN_LLR = "minLLR";
  public static final float DEFAULT_MIN_LLR = 1.0f;
  
  private long ngramTotal;
  private float minLLRValue;
  private boolean emitUnigrams;
  
  private final LLCallback ll;
  
  public LLRReducer() {
    this.ll = new ConcreteLLCallback();
  }
  
  /**
   * plug in an alternate LL implementation, used for testing
   * 
   * @param ll
   *          the LL to use.
   */
  LLRReducer(LLCallback ll) {
    this.ll = ll;
  }
  
  @Override
  public void configure(JobConf job) {
    super.configure(job);
    
    this.ngramTotal = job.getLong(NGRAM_TOTAL, -1);
    this.minLLRValue = job.getFloat(MIN_LLR, DEFAULT_MIN_LLR);
    
    this.emitUnigrams = job.getBoolean(CollocDriver.EMIT_UNIGRAMS, CollocDriver.DEFAULT_EMIT_UNIGRAMS);
    
    if (log.isInfoEnabled()) {
      log.info("NGram Total is {}", ngramTotal);
      log.info("Min LLR value is {}", minLLRValue);
      log.info("Emit Unitgrams is {}", emitUnigrams);
    }
    
    if (ngramTotal == -1) {
      throw new IllegalStateException("No NGRAM_TOTAL available in job config");
    }
  }
  
  /**
   * Perform LLR calculation, input is: k:ngram:ngramFreq v:(h_|t_)subgram:subgramfreq N = ngram total
   * 
   * Each ngram will have 2 subgrams, a head and a tail, referred to as A and B respectively below.
   * 
   * A+ B: number of times a+b appear together: ngramFreq A+!B: number of times A appears without B:
   * hSubgramFreq - ngramFreq !A+ B: number of times B appears without A: tSubgramFreq - ngramFreq !A+!B:
   * number of times neither A or B appears (in that order): N - (subgramFreqA + subgramFreqB - ngramFreq)
   */
  @Override
  public void reduce(Gram ngram,
                     Iterator<Gram> values,
                     OutputCollector<Text,DoubleWritable> output,
                     Reporter reporter) throws IOException {
    
    int[] gramFreq = new int[2];
    gramFreq[0] = gramFreq[1] = -1;
    
    if (ngram.getType() == Gram.Type.UNIGRAM && emitUnigrams) {
      DoubleWritable dd = new DoubleWritable(ngram.getFrequency());
      Text t = new Text(ngram.getString());
      output.collect(t, dd);
      return;
    }
    // FIXME: better way to handle errors? Wouldn't an exception thrown here
    // cause hadoop to re-try the job?
    String[] gram = new String[2];
    while (values.hasNext()) {
      Gram value = values.next();
      
      int pos = value.getType() == Gram.Type.HEAD ? 0 : 1;
      
      if (gramFreq[pos] != -1) {
        log.warn("Extra {} for {}, skipping", value.getType(), ngram);
        if (value.getType() == Gram.Type.HEAD) {
          reporter.incrCounter(Skipped.EXTRA_HEAD, 1);
        } else {
          reporter.incrCounter(Skipped.EXTRA_TAIL, 1);
        }
        return;
      }
      
      gram[pos] = value.getString();
      gramFreq[pos] = value.getFrequency();
    }
    
    if (gramFreq[0] == -1) {
      log.warn("Missing head for {}, skipping.", ngram);
      reporter.incrCounter(Skipped.MISSING_HEAD, 1);
      return;
    } else if (gramFreq[1] == -1) {
      log.warn("Missing tail for {}, skipping", ngram);
      reporter.incrCounter(Skipped.MISSING_TAIL, 1);
      return;
    }
    
    int k11 = ngram.getFrequency(); /* a&b */
    int k12 = gramFreq[0] - ngram.getFrequency(); /* a&!b */
    int k21 = gramFreq[1] - ngram.getFrequency(); /* !b&a */
    int k22 = (int) (ngramTotal - (gramFreq[0] + gramFreq[1] - ngram.getFrequency())); /* !a&!b */
    
    try {
      double llr = ll.logLikelihoodRatio(k11, k12, k21, k22);
      if (llr < minLLRValue) {
        reporter.incrCounter(Skipped.LESS_THAN_MIN_LLR, 1);
        return;
      }
      DoubleWritable dd = new DoubleWritable(llr);
      Text t = new Text(ngram.getString());
      output.collect(t, dd);
    } catch (IllegalArgumentException ex) {
      reporter.incrCounter(Skipped.LLR_CALCULATION_ERROR, 1);
      log.error("Problem calculating LLR ratio: " + ex.getMessage());
      log.error("NGram: " + ngram);
      log.error("HEAD: " + gram[0] + ':' + gramFreq[0]);
      log.error("TAIL: " + gram[1] + ':' + gramFreq[1]);
      log.error("k11: " + k11 + " k12: " + k12 + " k21: " + k21 + " k22: " + k22);
    }
  }
  
  /**
   * provide interface so the input to the llr calculation can be captured for validation in unit testing
   */
  public interface LLCallback {
    double logLikelihoodRatio(int k11, int k12, int k21, int k22);
  }
  
  /** concrete implementation delegates to LogLikelihood class */
  public static final class ConcreteLLCallback implements LLCallback {
    @Override
    public double logLikelihoodRatio(int k11, int k12, int k21, int k22) {
      return LogLikelihood.logLikelihoodRatio(k11, k12, k21, k22);
    }
  }
}
