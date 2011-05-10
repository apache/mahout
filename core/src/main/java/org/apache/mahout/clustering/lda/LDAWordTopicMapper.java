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

package org.apache.mahout.clustering.lda;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Runs inference on the input documents (which are sparse vectors of word counts) and outputs the sufficient
 * statistics for the word-topic assignments.
 */
public class LDAWordTopicMapper extends Mapper<WritableComparable<?>,VectorWritable,IntPairWritable,DoubleWritable> {
  
  private LDAState state;
  private LDAInference infer;
  
  @Override
  protected void map(WritableComparable<?> key,
                     VectorWritable wordCountsWritable,
                     Context context) throws IOException, InterruptedException {
    Vector wordCounts = wordCountsWritable.get();
    LDAInference.InferredDocument doc;
    try {
      doc = infer.infer(wordCounts);
    } catch (ArrayIndexOutOfBoundsException e1) {
      throw new IllegalStateException(
         "This is probably because the --numWords argument is set too small.  \n"
         + "\tIt needs to be >= than the number of words (terms actually) in the corpus and can be \n"
         + "\tlarger if some storage inefficiency can be tolerated.", e1);
    }
    
    double[] logTotals = new double[state.getNumTopics()];
    Arrays.fill(logTotals, Double.NEGATIVE_INFINITY);
    
    // Output sufficient statistics for each word. == pseudo-log counts.
    DoubleWritable v = new DoubleWritable();
    for (Iterator<Vector.Element> iter = wordCounts.iterateNonZero(); iter.hasNext();) {
      Vector.Element e = iter.next();
      int w = e.index();
      
      for (int k = 0; k < state.getNumTopics(); ++k) {
        v.set(doc.phi(k, w) + Math.log(e.get()));
        
        IntPairWritable kw = new IntPairWritable(k, w);
        
        // output (topic, word)'s logProb contribution
        context.write(kw, v);
        logTotals[k] = LDAUtil.logSum(logTotals[k], v.get());
      }
    }

    // Output the totals for the statistics. This is to make
    // normalizing a lot easier.
    for (int k = 0; k < state.getNumTopics(); ++k) {
      IntPairWritable kw = new IntPairWritable(k, LDADriver.TOPIC_SUM_KEY);
      v.set(logTotals[k]);
      assert !Double.isNaN(v.get());
      context.write(kw, v);
    }
    IntPairWritable llk = new IntPairWritable(LDADriver.LOG_LIKELIHOOD_KEY, LDADriver.LOG_LIKELIHOOD_KEY);
    // Output log-likelihoods.
    v.set(doc.getLogLikelihood());
    context.write(llk, v);
  }
  
  public void configure(LDAState myState) {
    this.state = myState;
    this.infer = new LDAInference(state);
  }
  
  public void configure(Configuration job) {
    LDAState myState = LDADriver.createState(job);
    configure(myState);
  }
  
  @Override
  protected void setup(Context context) {
    configure(context.getConfiguration());
  }
  
}
