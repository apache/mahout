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

import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.IndexSearcher;

import java.io.IOException;

/**
 * Utility for checking if a field exist in a Lucene index.
 */
public class LuceneIndexHelper {

  private LuceneIndexHelper() {

  }

  public static void fieldShouldExistInIndex(IndexSearcher searcher, String field) throws IOException {
    CollectionStatistics idFieldStatistics = searcher.collectionStatistics(field);
    if (idFieldStatistics.docCount() == 0) {
      throw new IllegalArgumentException("Field '" + field + "' does not exist in the index");
    }
  }

}
