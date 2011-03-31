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

package org.apache.mahout.cf.taste.impl.similarity.file;

import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.common.iterator.FileLineIterator;
import org.apache.mahout.common.iterator.TransformingIterator;

import java.io.File;
import java.io.IOException;
import java.util.regex.Pattern;

/**
 * a simple iterator using a {@link FileLineIterator} internally, parsing each
 * line into an {@link org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity.ItemItemSimilarity}
 */
final class FileItemItemSimilarityIterator
  extends TransformingIterator<String,GenericItemSimilarity.ItemItemSimilarity> {

  private static final Pattern SEPARATOR = Pattern.compile("[,\t]");

  FileItemItemSimilarityIterator(File similaritiesFile) throws IOException {
    super(new FileLineIterator(similaritiesFile));
  }

  @Override
  protected GenericItemSimilarity.ItemItemSimilarity transform(String in) {
    String[] tokens = SEPARATOR.split(in);
    return new GenericItemSimilarity.ItemItemSimilarity(Long.parseLong(tokens[0]),
                                                        Long.parseLong(tokens[1]),
                                                        Double.parseDouble(tokens[2]));
  }

}
