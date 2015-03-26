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

package org.apache.mahout.common.lucene;

import com.google.common.collect.AbstractIterator;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import java.io.IOException;

/**
 * Provide an Iterator for the tokens in a TokenStream.
 *
 * Note, it is the responsibility of the instantiating class to properly consume the
 * {@link org.apache.lucene.analysis.TokenStream}.  See the Lucene {@link org.apache.lucene.analysis.TokenStream}
 * documentation for more information.
 */
//TODO: consider using the char/byte arrays instead of strings, esp. when we upgrade to Lucene 4.0
public final class TokenStreamIterator extends AbstractIterator<String> {

  private final TokenStream tokenStream;

  public TokenStreamIterator(TokenStream tokenStream) {
    this.tokenStream = tokenStream;
  }

  @Override
  protected String computeNext() {
    try {
      if (tokenStream.incrementToken()) {
        return tokenStream.getAttribute(CharTermAttribute.class).toString();
      } else {
        tokenStream.end();
        tokenStream.close();
        return endOfData();
      }
    } catch (IOException e) {
      throw new IllegalStateException("IO error while tokenizing", e);
    }
  }

}
