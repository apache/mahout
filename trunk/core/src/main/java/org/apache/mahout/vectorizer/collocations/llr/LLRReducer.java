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

package org.apache.mahout.vectorizer.collocations.llr;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.stats.LogLikelihood;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reducer for pass 2 of the collocation discovery job. Collects ngram and sub-ngram frequencies and performs
 * the Log-likelihood ratio calculation.
 */
public class LLRReducer extends Reducer<Gram, Gram, Text, DoubleWritable> {

  /** Counter to track why a particlar entry was skipped */
  public enum Skipped {
    EXTRA_HEAD, EXTRA_TAIL, MISSING_HEAD, MISSING_TAIL, LESS_THAN_MIN_LLR, LLR_CALCULATION_ERROR,
  }

  private static final Logger log = LoggerFactory.getLogger(LLRReducer.class);

  public static final String NGRAM_TOTAL = "ngramTotal";
  public static final String MIN_LLR = "minLLR";
  public static final float DEFAULT_MIN_LLR = 1.0f;

  private long ngramTotal;
  private float minLLRValue;
  private boolean emitUnigrams;
  private final LLCallback ll;

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
  protected void reduce(Gram ngram, Iterable<Gram> values, Context context) throws IOException, InterruptedException {

    int[] gramFreq = {-1, -1};

    if (ngram.getType() == Gram.Type.UNIGRAM && emitUnigrams) {
      DoubleWritable dd = new DoubleWritable(ngram.getFrequency());
      Text t = new Text(ngram.getString());
      context.write(t, dd);
      return;
    }
    // TODO better way to handle errors? Wouldn't an exception thrown here
    // cause hadoop to re-try the job?
    String[] gram = new String[2];
    for (Gram value : values) {

      int pos = value.getType() == Gram.Type.HEAD ? 0 : 1;

      if (gramFreq[pos] != -1) {
        log.warn("Extra {} for {}, skipping", value.getType(), ngram);
        if (value.getType() == Gram.Type.HEAD) {
          context.getCounter(Skipped.EXTRA_HEAD).increment(1);
        } else {
          context.getCounter(Skipped.EXTRA_TAIL).increment(1);
        }
        return;
      }

      gram[pos] = value.getString();
      gramFreq[pos] = value.getFrequency();
    }

    if (gramFreq[0] == -1) {
      log.warn("Missing head for {}, skipping.", ngram);
      context.getCounter(Skipped.MISSING_HEAD).increment(1);
      return;
    } else if (gramFreq[1] == -1) {
      log.warn("Missing tail for {}, skipping", ngram);
      context.getCounter(Skipped.MISSING_TAIL).increment(1);
      return;
    }

    int k11 = ngram.getFrequency(); /* a&b */
    int k12 = gramFreq[0] - ngram.getFrequency(); /* a&!b */
    int k21 = gramFreq[1] - ngram.getFrequency(); /* !b&a */
    int k22 = (int) (ngramTotal - (gramFreq[0] + gramFreq[1] - ngram.getFrequency())); /* !a&!b */

    try {
      double llr = ll.logLikelihoodRatio(k11, k12, k21, k22);
      if (llr < minLLRValue) {
        context.getCounter(Skipped.LESS_THAN_MIN_LLR).increment(1);
        return;
      }
      DoubleWritable dd = new DoubleWritable(llr);
      Text t = new Text(ngram.getString());
      context.write(t, dd);
    } catch (IllegalArgumentException ex) {
      context.getCounter(Skipped.LLR_CALCULATION_ERROR).increment(1);
      log.error("Problem calculating LLR ratio: " + ex.getMessage());
      log.error("NGram: " + ngram);
      log.error("HEAD: " + gram[0] + ':' + gramFreq[0]);
      log.error("TAIL: " + gram[1] + ':' + gramFreq[1]);
      log.error("k11: " + k11 + " k12: " + k12 + " k21: " + k21 + " k22: " + k22);
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    this.ngramTotal = conf.getLong(NGRAM_TOTAL, -1);
    this.minLLRValue = conf.getFloat(MIN_LLR, DEFAULT_MIN_LLR);

    this.emitUnigrams = conf.getBoolean(CollocDriver.EMIT_UNIGRAMS, CollocDriver.DEFAULT_EMIT_UNIGRAMS);

    if (log.isInfoEnabled()) {
      log.info("NGram Total is {}", ngramTotal);
      log.info("Min LLR value is {}", minLLRValue);
      log.info("Emit Unitgrams is {}", emitUnigrams);
    }

    if (ngramTotal == -1) {
      throw new IllegalStateException("No NGRAM_TOTAL available in job config");
    }
  }

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
