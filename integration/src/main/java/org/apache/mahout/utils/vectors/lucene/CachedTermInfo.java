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

import com.google.common.collect.Maps;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;
import org.apache.mahout.utils.vectors.TermEntry;
import org.apache.mahout.utils.vectors.TermInfo;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;


/**
 * Caches TermEntries from a single field.  Materializes all values in the TermEnum to memory (much like FieldCache)
 */
public class CachedTermInfo implements TermInfo {

  private final Map<String, TermEntry> termEntries;
  private final String field;

  public CachedTermInfo(IndexReader reader, String field, int minDf, int maxDfPercent) throws IOException {
    this.field = field;
    Terms t = MultiFields.getTerms(reader, field);
    TermsEnum te = t.iterator(null);

    int numDocs = reader.numDocs();
    double percent = numDocs * maxDfPercent / 100.0;
    //Should we use a linked hash map so that we know terms are in order?
    termEntries = Maps.newLinkedHashMap();
    int count = 0;
    BytesRef text;
    while ((text = te.next()) != null) {
      int df = te.docFreq();
      if (df >= minDf && df <= percent) {
        TermEntry entry = new TermEntry(text.utf8ToString(), count++, df);
        termEntries.put(entry.getTerm(), entry);
      }
    }
  }

  @Override
  public int totalTerms(String field) {
    return termEntries.size();
  }

  @Override
  public TermEntry getTermEntry(String field, String term) {
    if (!this.field.equals(field)) {
      return null;
    }
    return termEntries.get(term);
  }

  @Override
  public Iterator<TermEntry> getAllEntries() {
    return termEntries.values().iterator();
  }
}
