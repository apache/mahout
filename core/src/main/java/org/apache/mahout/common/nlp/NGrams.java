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

package org.apache.mahout.common.nlp;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

public class NGrams {
  
  private final String line;
  
  private final int gramSize;
  
  public NGrams(String line, int gramSize) {
    this.line = line;
    this.gramSize = gramSize;
  }
  
  public Map<String,List<String>> generateNGrams() {
    Map<String,List<String>> returnDocument = Maps.newHashMap();
    
    StringTokenizer tokenizer = new StringTokenizer(line);
    List<String> tokens = Lists.newArrayList();
    String labelName = tokenizer.nextToken();
    List<String> previousN1Grams = Lists.newArrayList();
    while (tokenizer.hasMoreTokens()) {
      
      String nextToken = tokenizer.nextToken();
      if (previousN1Grams.size() == gramSize) {
        previousN1Grams.remove(0);
      }
      
      previousN1Grams.add(nextToken);
      
      StringBuilder gramBuilder = new StringBuilder();
      
      for (String gram : previousN1Grams) {
        gramBuilder.append(gram);
        String token = gramBuilder.toString();
        tokens.add(token);
        gramBuilder.append(' ');
      }
    }
    returnDocument.put(labelName, tokens);
    return returnDocument;
  }
  
  public List<String> generateNGramsWithoutLabel() {
    
    StringTokenizer tokenizer = new StringTokenizer(line);
    List<String> tokens = Lists.newArrayList();
    
    List<String> previousN1Grams = Lists.newArrayList();
    while (tokenizer.hasMoreTokens()) {
      
      String nextToken = tokenizer.nextToken();
      if (previousN1Grams.size() == gramSize) {
        previousN1Grams.remove(0);
      }
      
      previousN1Grams.add(nextToken);
      
      StringBuilder gramBuilder = new StringBuilder();
      
      for (String gram : previousN1Grams) {
        gramBuilder.append(gram);
        String token = gramBuilder.toString();
        tokens.add(token);
        gramBuilder.append(' ');
      }
    }
    
    return tokens;
  }
}
