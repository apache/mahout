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
import java.util.NoSuchElementException;

import com.google.common.base.Preconditions;
import org.apache.lucene.document.FieldSelector;
import org.apache.lucene.document.SetBasedFieldSelector;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.index.TermFreqVector;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;

/**
 * An {@link Iterator} over {@link Vector}s that uses a Lucene index as the source for creating the
 * {@link Vector}s. The field used to create the vectors currently must have term vectors stored for it.
 */
public final class LuceneIterator implements Iterator<Vector> {

  private final IndexReader indexReader;
  private final String field;
  private final String idField;
  private final FieldSelector idFieldSelector;
  private final VectorMapper mapper;
  private final double normPower;
  private final TermDocs termDocs;
  private Vector current;
  private boolean available;

  /**
   * Produce a LuceneIterable that can create the Vector plus normalize it.
   *
   * @param indexReader {@link org.apache.lucene.index.IndexReader} to read the documents from.
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
    // term docs(null) is a better way of iterating all the docs in Lucene
    Preconditions.checkArgument(normPower == LuceneIterable.NO_NORMALIZING || normPower >= 0,
                                "If specified normPower must be nonnegative", normPower);
    idFieldSelector = new SetBasedFieldSelector(Collections.singleton(idField), Collections.<String>emptySet());
    this.indexReader = indexReader;
    this.idField = idField;
    this.field = field;
    this.mapper = mapper;
    this.normPower = normPower;
    // term docs(null) is a better way of iterating all the docs in Lucene
    this.termDocs = indexReader.termDocs(null);
    current = null;
    available = false;
  }

  private void readVector() throws IOException {
    available = termDocs.next();
    if (!available) {
      current = null;
      return;
    }
    int doc = termDocs.doc();
    TermFreqVector termFreqVector = indexReader.getTermFreqVector(doc, field);
    if (termFreqVector == null) {
      throw new IllegalStateException("Field '" + field + "' does not have term vectors");
    }

    indexReader.getTermFreqVector(doc, field, mapper);
    mapper.setDocumentNumber(doc);
    Vector result = mapper.getVector();
    if (result == null) {
      // TODO is this right? last version would produce null in the iteration in this case, though it
      // seems like that may not be desirable
      current = null;
      return;
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
    current = result;
  }

  @Override
  public boolean hasNext() {
    if (!available) {
      try {
        readVector();
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    }
    return available;
  }

  @Override
  public Vector next() {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }
    Vector next = current;
    current = null;
    available = false;
    return next;
  }

  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }

}
