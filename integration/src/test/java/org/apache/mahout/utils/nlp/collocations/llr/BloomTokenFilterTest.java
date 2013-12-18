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
import java.io.Reader;
import java.io.StringReader;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetEncoder;

import com.google.common.base.Charsets;

import org.apache.hadoop.util.bloom.BloomFilter;
import org.apache.hadoop.util.bloom.Filter;
import org.apache.hadoop.util.bloom.Key;
import org.apache.hadoop.util.hash.Hash;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

public final class BloomTokenFilterTest extends MahoutTestCase {
  
  private static final CharsetEncoder encoder = Charsets.UTF_8.newEncoder();

  private static final String input = "The best of times the worst of times";
  private static final String[] allTokens = {
      "The", "best", "of", "times", "the", "worst", "of", "times"
  };
  private static final String[] expectedNonKeepTokens = { "best", "times", "the", "worst", "times" };
  private static final String[] expectedKeepTokens = { "The", "of", "of" };
  private static final String[] filterTokens    = { "The", "of" };
  private static final String[] notFilterTokens = { "best", "worst", "the", "times"};
  private static final String[] shingleKeepTokens = {
      "The best", "best of times", "the worst", "worst of times", "of times"
  };
  private static final String[] expectedShingleTokens = {
      "The best", "best of times", "of times", "the worst", "worst of times", "of times"
  };
  
  /** test standalone filter without tokenfilter wrapping */
  @Test
  public void testFilter() throws IOException {
    Filter filter = getFilter(filterTokens);
    Key k = new Key();
    for (String s: filterTokens) {
      setKey(k,s);
      assertTrue("Key for string " + s + " should be filter member", filter.membershipTest(k));
    }
    
    for (String s: notFilterTokens)  {
      setKey(k,s);
      assertFalse("Key for string " + s + " should not be filter member", filter.membershipTest(k));
    }
  }
  
  /** normal case, unfiltered analyzer */
  @Test
  public void testAnalyzer() throws IOException {
    Reader reader = new StringReader(input);
    Analyzer analyzer = new WhitespaceAnalyzer(Version.LUCENE_46);
    TokenStream ts = analyzer.tokenStream(null, reader);
    ts.reset();
    validateTokens(allTokens, ts);
    ts.end();
    ts.close();
  }
  
  /** filtered analyzer */
  @Test
  public void testNonKeepdAnalyzer() throws IOException {
    Reader reader = new StringReader(input);
    Analyzer analyzer = new WhitespaceAnalyzer(Version.LUCENE_46);
    TokenStream ts = analyzer.tokenStream(null, reader);
    ts.reset();
    TokenStream f = new BloomTokenFilter(getFilter(filterTokens), false /* toss matching tokens */, ts);
    validateTokens(expectedNonKeepTokens, f);
    ts.end();
    ts.close();
  }

  /** keep analyzer */
  @Test
  public void testKeepAnalyzer() throws IOException {
    Reader reader = new StringReader(input);
    Analyzer analyzer = new WhitespaceAnalyzer(Version.LUCENE_46);
    TokenStream ts = analyzer.tokenStream(null, reader);
    ts.reset();
    TokenStream f = new BloomTokenFilter(getFilter(filterTokens), true /* keep matching tokens */, ts);
    validateTokens(expectedKeepTokens, f);
    ts.end();
    ts.close();
  }
  
  /** shingles, keep those matching whitelist */
  @Test
  public void testShingleFilteredAnalyzer() throws IOException {
    Reader reader = new StringReader(input);
    Analyzer analyzer = new WhitespaceAnalyzer(Version.LUCENE_46);
    TokenStream ts = analyzer.tokenStream(null, reader);
    ts.reset();
    ShingleFilter sf = new ShingleFilter(ts, 3);
    TokenStream f = new BloomTokenFilter(getFilter(shingleKeepTokens),  true, sf);
    validateTokens(expectedShingleTokens, f);
    ts.end();
    ts.close();
  }
  
  private static void setKey(Key k, String s) throws IOException {
    ByteBuffer buffer = encoder.encode(CharBuffer.wrap(s.toCharArray()));
    k.set(buffer.array(), 1.0);
  }
  
  private static void validateTokens(String[] expected, TokenStream ts) throws IOException {
    int pos = 0;
    while (ts.incrementToken()) {
      assertTrue("Analyzer produced too many tokens", pos <= expected.length);
      CharTermAttribute termAttr = ts.getAttribute(CharTermAttribute.class);
      assertEquals("Unexpected term", expected[pos++], termAttr.toString());
    }
    assertEquals("Analyzer produced too few terms", expected.length, pos);
  }

  private static Filter getFilter(String[] tokens) throws IOException {
    Filter filter = new BloomFilter(100,50, Hash.JENKINS_HASH);
    Key k = new Key();
    for (String s: tokens) {
      setKey(k,s);
      filter.add(k);
    }
    return filter;
  }
  
}
