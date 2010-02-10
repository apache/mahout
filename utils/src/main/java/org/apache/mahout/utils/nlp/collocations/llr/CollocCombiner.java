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

import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.utils.nlp.collocations.llr.Gram.Type;

public class CollocCombiner extends MapReduceBase implements
Reducer<Gram, Gram, Gram, Gram> {

  /** collocation finder: pass 1 collec phase:
   *  
   *  given input from the mapper,
   *  k:h_subgram:1 v:ngram:1
   *  k:t_subgram:1 v:ngram:1
   *  
   *  count ngrams and subgrams. 
   *  
   *  output is:
   *  
   *  k:h_subgram:subgramfreq v:ngram:ngramfreq
   *  k:t_subgram:subgramfreq v:ngram:ngramfreq
   *  
   *  Each ngram's frequency is essentially counted twice, frequency should
   *  be the same for the head and tail. Fix this to count only for the head
   *  and move the count into the value?
   */
  @Override
  public void reduce(Gram key, Iterator<Gram> value,
      OutputCollector<Gram, Gram> output, Reporter reporter) throws IOException {

    HashMap<Gram,Gram> set = new HashMap<Gram,Gram>();
    int subgramFrequency = 0;

    while (value.hasNext()) {
      Gram t = value.next();
      subgramFrequency += t.getFrequency();

      Gram s = set.get(t);
      if (s == null) {
        // t is potentially reused, so create a new object to populate the HashMap
        Gram e = new Gram(t);
        set.put(e,e);
      }
      else {
        s.incrementFrequency(t.getFrequency());
      }
    }

    // emit subgram:subgramFreq ngram:ngramFreq pairs
    key.setFrequency(subgramFrequency);

    for (Gram t: set.keySet()) {
      if(key.getType() == Type.UNIGRAM)
        t.setType(key.getType());
      output.collect(key, t);
    }
  }

}
