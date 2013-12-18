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

package org.apache.mahout.utils.regex;

import com.google.common.io.Closeables;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.lucene.TokenStreamIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.StringReader;

public class AnalyzerTransformer implements RegexTransformer {

  private Analyzer analyzer;
  private String fieldName = "text";

  private static final Logger log = LoggerFactory.getLogger(AnalyzerTransformer.class);

  public AnalyzerTransformer() {
    this(new StandardAnalyzer(Version.LUCENE_46), "text");
  }

  public AnalyzerTransformer(Analyzer analyzer) {
    this(analyzer, "text");
  }

  public AnalyzerTransformer(Analyzer analyzer, String fieldName) {
    this.analyzer = analyzer;
    this.fieldName = fieldName;
  }

  @Override
  public String transformMatch(String match) {
    StringBuilder result = new StringBuilder();
    TokenStream ts = null;
    try {
      ts = analyzer.tokenStream(fieldName, new StringReader(match));
      ts.addAttribute(CharTermAttribute.class);
      ts.reset();
      TokenStreamIterator iter = new TokenStreamIterator(ts);
      while (iter.hasNext()) {
        result.append(iter.next()).append(' ');
      }
      ts.end();
    } catch (IOException e) {
      throw new IllegalStateException(e);
    } finally {
      try {
        Closeables.close(ts, true);
      } catch (IOException e) {
        log.error(e.getMessage(), e);
      }
    }
    return result.toString();
  }

  public Analyzer getAnalyzer() {
    return analyzer;
  }

  public void setAnalyzer(Analyzer analyzer) {
    this.analyzer = analyzer;
  }
}
