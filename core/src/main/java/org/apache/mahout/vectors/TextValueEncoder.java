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

package org.apache.mahout.vectors;

import com.google.common.base.Splitter;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Encodes text that is tokenized on non-alphanum separators.  Each word is encoded using a
 * settable encoder which is by default an StaticWordValueEncoder which gives all
 * words the same weight.
 */
public class TextValueEncoder extends FeatureVectorEncoder {

  private final Splitter onNonWord = Splitter.on(Pattern.compile("\\W+")).omitEmptyStrings();
  private FeatureVectorEncoder wordEncoder;
  private static final double LOG_2 = Math.log(2);

  public TextValueEncoder(String name) {
    super(name, 2);
    wordEncoder = new StaticWordValueEncoder(name);
  }

  /**
   * Adds a value to a vector after tokenizing it by splitting on non-alphanum characters.
   *
   * @param originalForm The original form of the value as a string.
   * @param data         The vector to which the value should be added.
   */
  @Override
  public void addToVector(String originalForm, double weight, Vector data) {
    Multiset<String> counts = HashMultiset.create();
    for (String word : tokenize(originalForm)) {
      counts.add(word);
    }
    for (String word : counts.elementSet()) {
      wordEncoder.addToVector(word, weight * Math.log(1 + counts.count(word))/LOG_2, data);
    }
  }

  @Override
  protected int hashForProbe(String originalForm, int dataSize, String name, int probe) {
    return 0;
  }

  @Override
  protected Iterable<Integer> hashesForProbe(String originalForm, int dataSize, String name, int probe){
    List<Integer> hashes = new ArrayList<Integer>();
    for (String word : tokenize(originalForm)){
      hashes.add(hashForProbe(word, dataSize, name, probe));
    }
    return hashes;
  }


  private Iterable<String> tokenize(CharSequence originalForm) {
    return onNonWord.split(originalForm);
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

  public void setWordEncoder(FeatureVectorEncoder wordEncoder) {
    this.wordEncoder = wordEncoder;
  }
}
