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

package org.apache.mahout.common.iterator;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

import com.google.common.base.Function;
import com.google.common.collect.ForwardingIterator;
import com.google.common.collect.Iterators;
import org.apache.mahout.common.Pair;

public class StringRecordIterator extends ForwardingIterator<Pair<List<String>,Long>> {
  
  private static final Long ONE = 1L;

  private final Pattern splitter;
  private final Iterator<Pair<List<String>,Long>> delegate;

  public StringRecordIterator(Iterable<String> stringIterator, String pattern) {
    this.splitter = Pattern.compile(pattern);
    delegate = Iterators.transform(
        stringIterator.iterator(),
        new Function<String,Pair<List<String>,Long>>() {
          @Override
          public Pair<List<String>,Long> apply(String from) {
            String[] items = splitter.split(from);
            return new Pair<List<String>,Long>(Arrays.asList(items), ONE);
          }
        });
  }

  @Override
  protected Iterator<Pair<List<String>,Long>> delegate() {
    return delegate;
  }

}
