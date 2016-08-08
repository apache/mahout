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

package org.apache.mahout.text.wikipedia;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopAnalyzer;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.analysis.util.CharArraySet;
import org.apache.lucene.analysis.util.StopwordAnalyzerBase;
import org.apache.lucene.analysis.wikipedia.WikipediaTokenizer;


public class WikipediaAnalyzer extends StopwordAnalyzerBase {
  
  public WikipediaAnalyzer() {
    super(StopAnalyzer.ENGLISH_STOP_WORDS_SET);
  }
  
  public WikipediaAnalyzer(CharArraySet stopSet) {
    super(stopSet);
  }

  @Override
  protected TokenStreamComponents createComponents(String fieldName) {
    Tokenizer tokenizer = new WikipediaTokenizer();
    TokenStream result = new StandardFilter(tokenizer);
    result = new LowerCaseFilter(result);
    result = new StopFilter(result, getStopwordSet());
    return new TokenStreamComponents(tokenizer, result);
  }
}
