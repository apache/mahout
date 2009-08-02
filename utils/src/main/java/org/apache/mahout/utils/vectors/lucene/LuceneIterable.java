package org.apache.mahout.utils.vectors.lucene;
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

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.TermDocs;
import org.apache.lucene.document.FieldSelector;
import org.apache.lucene.document.SetBasedFieldSelector;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.vectors.VectorIterable;

import java.io.IOException;
import java.util.Iterator;
import java.util.Collections;


/**
 *
 *
 **/
public class LuceneIterable implements VectorIterable {


  private IndexReader indexReader;
  private String field;
  private String idField;
  private FieldSelector idFieldSelector;

  private VectorMapper mapper;
  private double normPower = NO_NORMALIZING;

  public static final double NO_NORMALIZING = -1.0;

  public LuceneIterable(IndexReader reader, String idField, String field, VectorMapper mapper) {
    this(reader, idField, field, mapper, NO_NORMALIZING);
  }

  /**
   * Produce a LuceneIterable that can create the Vector plus normalize it.
   * @param reader The {@link org.apache.lucene.index.IndexReader} to read the documents from.
   * @param idField - The Field containing the id.  May be null
   * @param field The field to use for the Vector
   * @param mapper The {@link org.apache.mahout.utils.vectors.lucene.VectorMapper} for creating {@link org.apache.mahout.matrix.Vector}s from Lucene's TermVectors.
   * @param normPower The normalization value.  Must be greater than or equal to 0 or equal to {@link #NO_NORMALIZING}
   */
  public LuceneIterable(IndexReader reader, String idField, String field, VectorMapper mapper, double normPower) {
    if (normPower != NO_NORMALIZING && normPower < 0){
      throw new IllegalArgumentException("normPower must either be -1 or >= 0");
    }
      idFieldSelector = new SetBasedFieldSelector(Collections.singleton(idField), Collections.emptySet());
    this.indexReader = reader;
    this.idField = idField;
    this.field = field;
    this.mapper = mapper;
    this.normPower = normPower;
  }


  @Override
  public Iterator<Vector> iterator() {
    try {
      return new TDIterator();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private class TDIterator implements Iterator<Vector> {
    private TermDocs termDocs;

    private TDIterator() throws IOException {
      //term docs(null) is a better way of iterating all the docs in Lucene
      this.termDocs = indexReader.termDocs(null);
    }

    @Override
    public boolean hasNext() {
      try {
        return termDocs.next();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    @Override
    public Vector next() {
      Vector result = null;
      int doc = termDocs.doc();
      //
      try {
        indexReader.getTermFreqVector(doc, field, mapper);
        result = mapper.getVector();
        if (idField != null) {
          String id = indexReader.document(doc, idFieldSelector).get(idField);
          result.setName(id);
        } else {
          result.setName(String.valueOf(doc));
        }
        if (normPower != NO_NORMALIZING){
          result = result.normalize(normPower);
        }
      } catch (IOException e) {
        //Log?
        throw new RuntimeException(e);
      }

      return result;
    }


    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }

  }


}
