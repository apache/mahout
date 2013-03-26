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

import com.google.common.collect.AbstractIterator;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.Bump125;
import org.apache.mahout.utils.vectors.TermInfo;
import org.apache.mahout.vectorizer.Weight;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Iterate over a Lucene index, extracting term vectors.
 * Subclasses define how much information to retrieve from the Lucene index.
 */
public abstract class AbstractLuceneIterator extends AbstractIterator<Vector> {
  private static final Logger log = LoggerFactory.getLogger(LuceneIterator.class);
  protected final IndexReader indexReader;
  protected final String field;
  protected final TermInfo terminfo;
  protected final double normPower;
  protected final Weight weight;
  protected final Bump125 bump = new Bump125();
  protected int nextDocId;
  protected int maxErrorDocs;
  protected int numErrorDocs;
  protected long nextLogRecord = bump.increment();
  protected int skippedErrorMessages;

  public AbstractLuceneIterator(TermInfo terminfo, double normPower, IndexReader indexReader, Weight weight,
      double maxPercentErrorDocs, String field) {
    this.terminfo = terminfo;
    this.normPower = normPower;
    this.indexReader = indexReader;

    this.weight = weight;
    this.nextDocId = 0;
    this.maxErrorDocs = (int) (maxPercentErrorDocs * indexReader.numDocs());
    this.field = field;
  }

  /**
   * Given the document name, derive a name for the vector. This may involve
   * reading the document from Lucene and setting up any other state that the
   * subclass wants. This will be called once for each document that the
   * iterator processes.
   * @param documentIndex the lucene document index.
   * @return the name to store in the vector.
   */
  protected abstract String getVectorName(int documentIndex) throws IOException;

  @Override
  protected Vector computeNext() {
    try {
      int doc;
      Terms termFreqVector;
      String name;

      do {
        doc = this.nextDocId;
        nextDocId++;

        if (doc >= indexReader.maxDoc()) {
          return endOfData();
        }

        termFreqVector = indexReader.getTermVector(doc, field);
        name = getVectorName(doc);

        if (termFreqVector == null) {
          numErrorDocs++;
          if (numErrorDocs >= maxErrorDocs) {
            log.error("There are too many documents that do not have a term vector for {}", field);
            throw new IllegalStateException("There are too many documents that do not have a term vector for "
                + field);
          }
          if (numErrorDocs >= nextLogRecord) {
            if (skippedErrorMessages == 0) {
              log.warn("{} does not have a term vector for {}", name, field);
            } else {
              log.warn("{} documents do not have a term vector for {}", numErrorDocs, field);
            }
            nextLogRecord = bump.increment();
            skippedErrorMessages = 0;
          } else {
            skippedErrorMessages++;
          }
        }
      } while (termFreqVector == null);

      // The loop exits with termFreqVector and name set.

      TermsEnum te = termFreqVector.iterator(null);
      BytesRef term;
      TFDFMapper mapper = new TFDFMapper(indexReader.numDocs(), weight, this.terminfo);
      mapper.setExpectations(field, termFreqVector.size());
      while ((term = te.next()) != null) {
        mapper.map(term, (int) te.totalTermFreq());
      }
      Vector result = mapper.getVector();
      if (result == null) {
        // TODO is this right? last version would produce null in the iteration in this case, though it
        // seems like that may not be desirable
        return null;
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
