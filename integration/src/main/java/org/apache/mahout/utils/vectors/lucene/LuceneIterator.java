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

import com.google.common.base.Preconditions;
import com.google.common.collect.Sets;
import org.apache.lucene.index.IndexReader;
import org.apache.mahout.utils.vectors.TermInfo;
import org.apache.mahout.vectorizer.Weight;

import java.io.IOException;
import java.util.Set;

/**
 * An {@link java.util.Iterator} over {@link org.apache.mahout.math.Vector}s that uses a Lucene index as the source
 * for creating the {@link org.apache.mahout.math.Vector}s. The field used to create the vectors currently must have
 * term vectors stored for it.
 */
public class LuceneIterator extends AbstractLuceneIterator {

  protected final Set<String> idFieldSelector;
  protected final String idField;

    /**
   * Produce a LuceneIterable that can create the Vector plus normalize it.
   *
   * @param indexReader {@link IndexReader} to read the documents from.
   * @param idField     field containing the id. May be null.
   * @param field       field to use for the Vector
   * @param termInfo    termInfo
   * @param weight      weight
   * @param normPower   the normalization value. Must be non-negative, or {@link LuceneIterable#NO_NORMALIZING}
   */
  public LuceneIterator(IndexReader indexReader, String idField, String field, TermInfo termInfo, Weight weight,
                        double normPower) {
    this(indexReader, idField, field, termInfo, weight, normPower, 0.0);
  }

  /**
   * @param indexReader {@link IndexReader} to read the documents from.
   * @param idField    field containing the id. May be null.
   * @param field      field to use for the Vector
   * @param termInfo   termInfo
   * @param weight     weight
   * @param normPower  the normalization value. Must be non-negative, or {@link LuceneIterable#NO_NORMALIZING}
   * @param maxPercentErrorDocs most documents that will be tolerated without a term freq vector. In [0,1].
   * @see #LuceneIterator(org.apache.lucene.index.IndexReader, String, String, org.apache.mahout.utils.vectors.TermInfo,
   * org.apache.mahout.vectorizer.Weight, double)
   */
  public LuceneIterator(IndexReader indexReader,
                        String idField,
                        String field,
                        TermInfo termInfo,
                        Weight weight,
                        double normPower,
                        double maxPercentErrorDocs) {
      super(termInfo, normPower, indexReader, weight, maxPercentErrorDocs, field);
      // term docs(null) is a better way of iterating all the docs in Lucene
    Preconditions.checkArgument(normPower == LuceneIterable.NO_NORMALIZING || normPower >= 0,
        "normPower must be non-negative or -1, but normPower = " + normPower);
    Preconditions.checkArgument(maxPercentErrorDocs >= 0.0 && maxPercentErrorDocs <= 1.0,
        "Must be: 0.0 <= maxPercentErrorDocs <= 1.0");
    this.idField = idField;
    if (idField != null) {
      idFieldSelector = Sets.newTreeSet();
      idFieldSelector.add(idField);
    } else {
      /*The field in the index  containing the index. If null, then the Lucene internal doc id is used
      which is prone to error if the underlying index changes*/
      idFieldSelector = null;
    }
  }

  @Override
  protected String getVectorName(int documentIndex) throws IOException {
    String name;
    if (idField != null) {
      name = indexReader.document(documentIndex, idFieldSelector).get(idField);
    } else {
      name = String.valueOf(documentIndex);
    }
    return name;
  }
}
