/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.mahout.utils.nlp.collocations.llr;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CodingErrorAction;

import com.google.common.base.Charsets;
import org.apache.hadoop.util.bloom.Filter;
import org.apache.hadoop.util.bloom.Key;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

/**
 * Emits tokens based on bloom filter membership.
 */
public final class BloomTokenFilter extends TokenFilter {
  
  private final Filter filter;
  private final CharTermAttribute termAtt;
  private final CharsetEncoder encoder;
  private final Key key;
  private final boolean keepMembers;
  
  /** 
   * @param filter tokens will be checked for membership in this bloom filter
   * @param in the tokenstream to read.
   * @param keepMembers keep memoers of the bloom filter? If true works like
   *   a whitelist and members found in the list are kept and all others are
   *   dropped. If false works like a stoplist and members found in the 
   *   filter are dropped all others are kept.
   */
  public BloomTokenFilter(Filter filter, boolean keepMembers, TokenStream in) {
    super(in);
    this.filter = filter;
    this.keepMembers = keepMembers;
    this.key = new Key();
    this.termAtt = addAttribute(CharTermAttribute.class);
    this.encoder = Charsets.UTF_8.newEncoder().
      onMalformedInput(CodingErrorAction.REPORT).
      onUnmappableCharacter(CodingErrorAction.REPORT);
  }
  
  @Override
  public boolean incrementToken() throws IOException {
    while (input.incrementToken()) {
      ByteBuffer bytes =  encoder.encode(CharBuffer.wrap(termAtt.buffer(), 0, termAtt.length()));
      key.set(bytes.array(), 1.0f);
      boolean member = filter.membershipTest(key);
      if ((keepMembers && member) || (!keepMembers && !member)) {
        return true;
      }
    }
    return false;
  }

}
