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

import java.io.IOException;
import java.util.Collections;
import java.util.Iterator;

import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;
import org.apache.lucene.document.FieldSelector;
import org.apache.lucene.document.SetBasedFieldSelector;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.index.TermFreqVector;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.Bump125;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An {@link Iterator} over {@link Vector}s that uses a Lucene index as the source for creating the
 * {@link Vector}s. The field used to create the vectors currently must have term vectors stored for it.
 */
public final class LuceneIterator extends AbstractIterator<Vector> {

  private static final Logger log = LoggerFactory.getLogger(LuceneIterator.class);

  private final IndexReader indexReader;
  private final String field;
  private final String idField;
  private final FieldSelector idFieldSelector;
  private final VectorMapper mapper;
  private final double normPower;
  private final TermDocs termDocs;

  private int numErrorDocs = 0;
  private int maxErrorDocs = 0;
  private final Bump125 bump = new Bump125();
  private long nextLogRecord = bump.increment();
  private int skippedErrorMessages = 0;

  /**
   * Produce a LuceneIterable that can create the Vector plus normalize it.
   *
   * @param indexReader {@link IndexReader} to read the documents from.
   * @param idField field containing the id. May be null.
   * @param field  field to use for the Vector
   * @param mapper {@link VectorMapper} for creating {@link Vector}s from Lucene's TermVectors.
   * @param normPower the normalization value. Must be nonnegative, or {@link LuceneIterable#NO_NORMALIZING}
   */
  public LuceneIterator(IndexReader indexReader,
                        String idField,
                        String field,
                        VectorMapper mapper,
                        double normPower) throws IOException {
    this(indexReader, idField, field, mapper, normPower, 0.0);
  }

  /**
   * @see #LuceneIterator(IndexReader, String, String, VectorMapper, double)
   * @param maxPercentErrorDocs most documents that will be tolerated without a term freq vector. In [0,1].
   */
  public LuceneIterator(IndexReader indexReader,
                        String idField,
                        String field,
                        VectorMapper mapper,
                        double normPower,
                        double maxPercentErrorDocs) throws IOException {
    // term docs(null) is a better way of iterating all the docs in Lucene
    Preconditions.checkArgument(normPower == LuceneIterable.NO_NORMALIZING || normPower >= 0,
                                "If specified normPower must be nonnegative", normPower);
    Preconditions.checkArgument(maxPercentErrorDocs >= 0.0 && maxPercentErrorDocs <= 1.0);
    idFieldSelector = new SetBasedFieldSelector(Collections.singleton(idField), Collections.<String>emptySet());
    this.indexReader = indexReader;
    this.idField = idField;
    this.field = field;
    this.mapper = mapper;
    this.normPower = normPower;
    // term docs(null) is a better way of iterating all the docs in Lucene
    this.termDocs = indexReader.termDocs(null);
    this.maxErrorDocs = (int) (maxPercentErrorDocs * indexReader.numDocs());
  }

  @Override
  protected Vector computeNext() {
    try {
      if (!termDocs.next()) {
        return endOfData();
      }

      int doc = termDocs.doc();
      TermFreqVector termFreqVector = indexReader.getTermFreqVector(doc, field);
      if (termFreqVector == null) {
        numErrorDocs++;
        if (numErrorDocs >= maxErrorDocs) {
          log.error("There are too many documents that do not have a term vector for {}", field);
          throw new IllegalStateException("There are too many documents that do not have a term vector for " + field);
        }
        if (numErrorDocs >= nextLogRecord) {
          if (skippedErrorMessages == 0) {
            log.warn("{} does not have a term vector for {}", indexReader.document(doc).get(idField), field);
          } else {
            log.warn("{} documents do not have a term vector for {}", numErrorDocs, field);
          }
          nextLogRecord = bump.increment();
          skippedErrorMessages = 0;
        } else {
          skippedErrorMessages++;
        }
        computeNext();
      }

      indexReader.getTermFreqVector(doc, field, mapper);
      mapper.setDocumentNumber(doc);
      Vector result = mapper.getVector();
      if (result == null) {
        // TODO is this right? last version would produce null in the iteration in this case, though it
        // seems like that may not be desirable
        return null;
      }
      String name;
      if (idField != null) {
        name = indexReader.document(doc, idFieldSelector).get(idField);
      } else {
        name = String.valueOf(doc);
      }
      if (normPower == LuceneIterable.NO_NORMALIZING) {
        result = new NamedVector(result, name);
      } else {
        result = new NamedVector(result.normalize(normPower), name);
      }
      return result;
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
  }

}
