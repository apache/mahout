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

package org.apache.mahout.common.lucene;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import java.util.Iterator;

/** Used to emit tokens from an input string array in the style of TokenStream */
public final class IteratorTokenStream extends TokenStream {
  private final CharTermAttribute termAtt;
  private final Iterator<String> iterator;

  public IteratorTokenStream(Iterator<String> iterator) {
    this.iterator = iterator;
    this.termAtt = addAttribute(CharTermAttribute.class);
  }

  @Override
  public boolean incrementToken() {
    if (iterator.hasNext()) {
      clearAttributes();
      termAtt.append(iterator.next());
      return true;
    } else {
      return false;
    }
  }
}
