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
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;

/**
 * Used for testing lucene2seq
 */
public class SingleFieldDocument {

  public static final String ID_FIELD = "idField";
  public static final String FIELD = "field";

  private String id;
  private String field;

  public SingleFieldDocument(String id, String field) {
    this.id = id;
    this.field = field;
  }

  public String getId() {
    return id;
  }

  public String getField() {
    return field;
  }

  public Document asLuceneDocument() {
    Document document = new Document();

    Field idField = new StringField(ID_FIELD, getId(), Field.Store.YES);
    Field field = new TextField(FIELD, getField(), Field.Store.YES);

    document.add(idField);
    document.add(field);

    return document;
  }
}
