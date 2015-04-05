package org.apache.mahout.text;
/*
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
import org.apache.lucene.index.IndexableField;

import java.io.IOException;

/**
 * Utility for checking if a field is stored in a Lucene index.
 */
public class LuceneIndexHelper {

  private LuceneIndexHelper() {

  }

  public static void fieldShouldExistInIndex(IndexReader reader, String fieldName) throws IOException {
    IndexableField field = reader.document(0).getField(fieldName);
    if (field == null || !field.fieldType().stored()) {
      throw new IllegalArgumentException("Field '" + fieldName +
          "' is possibly not stored since first document in index does not contain this field.");
    }
  }

}
