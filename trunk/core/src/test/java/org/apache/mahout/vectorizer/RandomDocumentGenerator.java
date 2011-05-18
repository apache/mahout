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

package org.apache.mahout.vectorizer;

import java.util.Random;

import org.apache.mahout.common.RandomUtils;

public class RandomDocumentGenerator {
  
  private static final int AVG_DOCUMENT_LENGTH = 20;
  private static final int AVG_SENTENCE_LENGTH = 8;
  private static final int AVG_WORD_LENGTH = 6;
  private static final String CHARSET = "abcdef";
  private static final String DELIM = " .,?;:!\t\n\r";
  private static final String ERRORSET = "`1234567890" + "-=~@#$%^&*()_+[]{}'\"/<>|\\";

  private final Random random = RandomUtils.getRandom();
  
  private char getRandomDelimiter() {
    return DELIM.charAt(random.nextInt(DELIM.length()));
  }

  public String getRandomDocument() {
    int length = (AVG_DOCUMENT_LENGTH >> 1) + random.nextInt(AVG_DOCUMENT_LENGTH);
    StringBuilder sb = new StringBuilder(length * AVG_SENTENCE_LENGTH * AVG_WORD_LENGTH);
    for (int i = 0; i < length; i++) {
      sb.append(getRandomSentence());
    }
    return sb.toString();
  }

  public String getRandomSentence() {
    int length = (AVG_SENTENCE_LENGTH >> 1) + random.nextInt(AVG_SENTENCE_LENGTH);
    StringBuilder sb = new StringBuilder(length * AVG_WORD_LENGTH);
    for (int i = 0; i < length; i++) {
      sb.append(getRandomString()).append(' ');
    }
    sb.append(getRandomDelimiter());
    return sb.toString();
  }

  public String getRandomString() {
    int length = (AVG_WORD_LENGTH >> 1) + random.nextInt(AVG_WORD_LENGTH);
    StringBuilder sb = new StringBuilder(length);
    for (int i = 0; i < length; i++) {
      sb.append(CHARSET.charAt(random.nextInt(CHARSET.length())));
    }
    if (random.nextInt(10) == 0) {
      sb.append(ERRORSET.charAt(random.nextInt(ERRORSET.length())));
    }
    return sb.toString();
  }
}
