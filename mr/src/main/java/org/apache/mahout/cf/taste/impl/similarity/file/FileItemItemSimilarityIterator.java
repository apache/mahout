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

import com.google.common.base.Function;
import com.google.common.collect.ForwardingIterator;
import com.google.common.collect.Iterators;
import org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity;
import org.apache.mahout.common.iterator.FileLineIterator;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.regex.Pattern;

/**
 * a simple iterator using a {@link FileLineIterator} internally, parsing each
 * line into an {@link GenericItemSimilarity.ItemItemSimilarity}.
 */
final class FileItemItemSimilarityIterator extends ForwardingIterator<GenericItemSimilarity.ItemItemSimilarity> {

  private static final Pattern SEPARATOR = Pattern.compile("[,\t]");

  private final Iterator<GenericItemSimilarity.ItemItemSimilarity> delegate;

  FileItemItemSimilarityIterator(File similaritiesFile) throws IOException {
    delegate = Iterators.transform(
        new FileLineIterator(similaritiesFile),
        new Function<String, GenericItemSimilarity.ItemItemSimilarity>() {
          @Override
          public GenericItemSimilarity.ItemItemSimilarity apply(String from) {
            String[] tokens = SEPARATOR.split(from);
            return new GenericItemSimilarity.ItemItemSimilarity(Long.parseLong(tokens[0]),
                                                                Long.parseLong(tokens[1]),
                                                                Double.parseDouble(tokens[2]));
          }
        });
  }

  @Override
  protected Iterator<GenericItemSimilarity.ItemItemSimilarity> delegate() {
    return delegate;
  }

}
