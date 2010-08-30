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
    for (String word : tokenize(originalForm)) {
      wordEncoder.addToVector(word, weight, data);
    }
  }

  @Override
  protected int hashForProbe(String originalForm, Vector data, String name, int i) {
    return 0;
  }

  protected Iterable<Integer> hashesForProbe(String originalForm, Vector data, String name, int i){
    List<Integer> hashes = new ArrayList<Integer>();
    for (String word : tokenize(originalForm)){
      hashes.add(hashForProbe(word,data,name,i));
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
