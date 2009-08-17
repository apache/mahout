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
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.MapContext;
import org.apache.mahout.matrix.AbstractVector;
import org.apache.mahout.matrix.Vector;

/**
* Runs inference on the input documents (which are
* sparse vectors of word counts) and outputs
* the sufficient statistics for the word-topic
* assignments.
*/
public class LDAMapper extends 
    Mapper<WritableComparable<?>, Vector, IntPairWritable, DoubleWritable> {

  private LDAState state;
  private LDAInference infer;

  @Override
  public void map(WritableComparable<?> key, Vector wordCounts, Context context)
      throws IOException, InterruptedException {
    LDAInference.InferredDocument doc = infer.infer(wordCounts);

    double[] logTotals = new double[state.numTopics];
    Arrays.fill(logTotals, Double.NEGATIVE_INFINITY);

    // Output sufficient statistics for each word. == pseudo-log counts.
    IntPairWritable kw = new IntPairWritable();
    DoubleWritable v = new DoubleWritable();
    for (Iterator<Vector.Element> iter = wordCounts.iterateNonZero();
        iter.hasNext();) {
      Vector.Element e = iter.next();
      int w = e.index();
      kw.setY(w);
      for (int k = 0; k < state.numTopics; ++k) {
        v.set(doc.phi(k, w) + Math.log(e.get()));

        kw.setX(k);

        // ouput (topic, word)'s logProb contribution
        context.write(kw, v);
        logTotals[k] = LDAUtil.logSum(logTotals[k], v.get());
      }
    }
    
    // Output the totals for the statistics. This is to make
    // normalizing a lot easier.
    kw.setY(LDADriver.TOPIC_SUM_KEY);
    for (int k = 0; k < state.numTopics; ++k) {
      kw.setX(k);
      v.set(logTotals[k]);
      assert !Double.isNaN(v.get());
      context.write(kw, v);
    }

    // Output log-likelihoods.
    kw.setX(LDADriver.LOG_LIKELIHOOD_KEY);
    kw.setY(LDADriver.LOG_LIKELIHOOD_KEY);
    v.set(doc.logLikelihood);
    context.write(kw, v);
  }

  public void configure(LDAState myState) {
    this.state = myState;
    this.infer = new LDAInference(state);
  }

  public void configure(Configuration job) {
    try {
      LDAState myState = LDADriver.createState(job);
      configure(myState);
    } catch (IOException e) {
      throw new RuntimeException("Error creating LDA State!", e);
    }
  }

  @Override
  protected void setup(Context context) {
    configure(context.getConfiguration());
  }


}
