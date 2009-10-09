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

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.TermVectorOffsetInfo;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.vectors.Weight;
import org.apache.mahout.utils.vectors.TermEntry;
import org.apache.mahout.utils.vectors.TermInfo;


/**
 * Not thread-safe
 */
public class TFDFMapper extends VectorMapper {

  //public static final int DEFAULT_CACHE_SIZE = 256;

  private final IndexReader reader; // TODO never used?
  private Vector vector;

  private final Weight weight;
  private int numTerms;
  private final TermInfo termInfo;
  private String field;
  private final int numDocs;

  public TFDFMapper(IndexReader reader, Weight weight, TermInfo termInfo) {
    this.reader = reader;
    this.weight = weight;
    this.termInfo = termInfo;
    this.numDocs = reader.numDocs();
  }

  @Override
  public Vector getVector() {
    return vector;
  }

  @Override
  public void setExpectations(String field, int numTerms, boolean storeOffsets, boolean storePositions) {
    this.field = field;
    vector = new SparseVector(termInfo.totalTerms(field));
    this.numTerms = numTerms;
  }

  @Override
  public void map(String term, int frequency, TermVectorOffsetInfo[] offsets, int[] positions) {
    TermEntry entry = termInfo.getTermEntry(field, term);
    if (entry != null) {
      vector.setQuick(entry.termIdx, weight.calculate(frequency, entry.docFreq, numTerms, numDocs));
    }
  }

  @Override
  public boolean isIgnoringPositions() {
    return true;
  }

  @Override
  public boolean isIgnoringOffsets() {
    return true;
  }
}
