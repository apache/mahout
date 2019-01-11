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

package org.apache.mahout.utils.vectors.lucene;

import org.apache.lucene.util.BytesRef;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.vectors.TermEntry;
import org.apache.mahout.utils.vectors.TermInfo;
import org.apache.mahout.vectorizer.Weight;


/**
 * Not thread-safe
 */
public class TFDFMapper  {

  private Vector vector;
  
  private final Weight weight;
  private long numTerms;
  private final TermInfo termInfo;
  private String field;
  private final int numDocs;
  
  public TFDFMapper(int numDocs, Weight weight, TermInfo termInfo) {
    this.weight = weight;
    this.termInfo = termInfo;
    this.numDocs = numDocs;
  }

  public void setExpectations(String field, long numTerms) {
    this.field = field;
    vector = new RandomAccessSparseVector(termInfo.totalTerms(field));
    this.numTerms = numTerms;
  }
  
  public void map(BytesRef term, int frequency) {
    TermEntry entry = termInfo.getTermEntry(field, term.utf8ToString());
    if (entry != null) {
      vector.setQuick(entry.getTermIdx(), weight.calculate(frequency, entry.getDocFreq(), (int)numTerms, numDocs));
    }
  }
  
  public Vector getVector() {
    return this.vector;
  }
  
}
