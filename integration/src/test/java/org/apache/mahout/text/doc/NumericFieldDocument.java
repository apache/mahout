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
package org.apache.mahout.text.doc;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;


/**
 * Document with numeric field.
 */
public class NumericFieldDocument extends SingleFieldDocument {

  public static final String NUMERIC_FIELD = "numeric";

  private int numericField;

  public NumericFieldDocument(String id, String field, int numericField) {
    super(id, field);
    this.numericField = numericField;
  }

  @Override
  public Document asLuceneDocument() {
    Document document = new Document();

    document.add(new StringField(ID_FIELD, getId(), Field.Store.YES));
    document.add(new TextField(FIELD, getField(), Field.Store.YES));
    document.add(new IntField(NUMERIC_FIELD, numericField, Field.Store.YES));

    return document;
  }

  public int getNumericField() {
    return numericField;
  }
}
