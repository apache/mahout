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

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.lucene.index.IndexFileNames;

import java.util.regex.Pattern;

/**
 * A wrapper class to convert an IndexFileNameFilter which implements
 * java.io.FilenameFilter to an org.apache.hadoop.fs.PathFilter.
 */
final class LuceneIndexFileNameFilter implements PathFilter {

  private static final LuceneIndexFileNameFilter LUCENE_INDEX_FILE_NAME_FILTER = new LuceneIndexFileNameFilter();

  /**
   * Get a static instance.
   *
   * @return the static instance
   */
  public static LuceneIndexFileNameFilter getFilter() {
    return LUCENE_INDEX_FILE_NAME_FILTER;
  }

  private LuceneIndexFileNameFilter() {}

  //TODO: Lucene defines this in IndexFileNames, but it is package private,
  // so make sure it doesn't change w/ new releases.
  private static final Pattern CODEC_FILE_PATTERN = Pattern.compile("_[a-z0-9]+(_.*)?\\..*");

  public boolean accept(Path path) {
    String name = path.getName();
    if (CODEC_FILE_PATTERN.matcher(name).matches() || name.startsWith(IndexFileNames.SEGMENTS)) {
      return true;
    }
    for (String extension : IndexFileNames.INDEX_EXTENSIONS) {
      if (name.endsWith(extension)) {
        return true;
      }
    }
    return false;
  }

}
