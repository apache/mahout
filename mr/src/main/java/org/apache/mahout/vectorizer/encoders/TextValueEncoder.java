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

package org.apache.mahout.vectorizer.encoders;

import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import org.apache.mahout.math.Vector;

import java.util.Collection;
import java.util.regex.Pattern;

/**
 * Encodes text that is tokenized on non-alphanum separators.  Each word is encoded using a
 * settable encoder which is by default an StaticWordValueEncoder which gives all
 * words the same weight.
 * @see LuceneTextValueEncoder
 */
public class TextValueEncoder extends FeatureVectorEncoder {

  private static final double LOG_2 = Math.log(2.0);

  private static final Splitter ON_NON_WORD = Splitter.on(Pattern.compile("\\W+")).omitEmptyStrings();

  private FeatureVectorEncoder wordEncoder;
  private final Multiset<String> counts;

  public TextValueEncoder(String name) {
    super(name, 2);
    wordEncoder = new StaticWordValueEncoder(name);
    counts = HashMultiset.create();
  }

  /**
   * Adds a value to a vector after tokenizing it by splitting on non-alphanum characters.
   *
   * @param originalForm The original form of the value as a string.
   * @param data         The vector to which the value should be added.
   */
  @Override
  public void addToVector(byte[] originalForm, double weight, Vector data) {
    addText(originalForm);
    flush(weight, data);
  }

  /**
   * Adds text to the internal word counter, but delays converting it to vector
   * form until flush is called.
   * @param originalForm  The original text encoded as UTF-8
   */
  public void addText(byte[] originalForm) {
    addText(new String(originalForm, Charsets.UTF_8));
  }

  /**
   * Adds text to the internal word counter, but delays converting it to vector
   * form until flush is called.
   * @param text  The original text encoded as UTF-8
   */
  public void addText(CharSequence text) {
    for (String word : tokenize(text)) {
      counts.add(word);
    }
  }

  /**
   * Adds all of the tokens that we counted up to a vector.
   */
  public void flush(double weight, Vector data) {
    for (String word : counts.elementSet()) {
      // weight words by log_2(tf) times whatever other weight we are given
      wordEncoder.addToVector(word, weight * Math.log1p(counts.count(word)) / LOG_2, data);
    }
    counts.clear();
  }

  @Override
  protected int hashForProbe(byte[] originalForm, int dataSize, String name, int probe) {
    return 0;
  }

  @Override
  protected Iterable<Integer> hashesForProbe(byte[] originalForm, int dataSize, String name, int probe) {
    Collection<Integer> hashes = Lists.newArrayList();
    for (String word : tokenize(new String(originalForm, Charsets.UTF_8))) {
      hashes.add(hashForProbe(bytesForString(word), dataSize, name, probe));
    }
    return hashes;
  }

  /**
   * Tokenizes a string using the simplest method.  This should be over-ridden for more subtle
   * tokenization.
   * @see LuceneTextValueEncoder
   */
  protected Iterable<String> tokenize(CharSequence originalForm) {
    return ON_NON_WORD.split(originalForm);
  }

  /**
   * Converts a value into a form that would help a human understand the internals of how the value
   * is being interpreted.  For text-like things, this is likely to be a list of the terms found with
   * associated weights (if any).
   *
   * @param originalForm The original form of the value as a string.
   * @return A string that a human can read.
   */
  @Override
  public String asString(String originalForm) {
    StringBuilder r = new StringBuilder();
    r.append('[');
    for (String word : tokenize(originalForm)) {
      if (r.length() > 1) {
        r.append(", ");
      }
      r.append(wordEncoder.asString(word));
    }
    r.append(']');
    return r.toString();
  }

  public final void setWordEncoder(FeatureVectorEncoder wordEncoder) {
    this.wordEncoder = wordEncoder;
  }
}
