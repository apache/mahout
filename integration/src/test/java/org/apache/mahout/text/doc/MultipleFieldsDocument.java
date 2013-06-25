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
import org.apache.lucene.document.TextField;

/**
 * Used for testing lucene2seq
 */
public class MultipleFieldsDocument extends SingleFieldDocument {

  public static final String FIELD1 = "field1";
  public static final String FIELD2 = "field2";

  private String field1;
  private String field2;

  public MultipleFieldsDocument(String id, String field, String field1, String field2) {
    super(id, field);
    this.field1 = field1;
    this.field2 = field2;
  }

  public String getField1() {
    return field1;
  }

  public String getField2() {
    return field2;
  }

  @Override
  public Document asLuceneDocument() {
    Document document = super.asLuceneDocument();

    document.add(new TextField(FIELD1, this.field1, Field.Store.YES));
    document.add(new TextField(FIELD2, this.field2, Field.Store.YES));

    return document;
  }
}
