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

package org.apache.mahout.classifier;

import com.google.common.base.Charsets;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ConstantValueEncoder;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.text.SimpleDateFormat;
import java.util.Collection;
import java.util.Date;
import java.util.Locale;
import java.util.Random;

public final class NewsgroupHelper {
  
  private static final SimpleDateFormat[] DATE_FORMATS = {
    new SimpleDateFormat("", Locale.ENGLISH),
    new SimpleDateFormat("MMM-yyyy", Locale.ENGLISH),
    new SimpleDateFormat("dd-MMM-yyyy HH:mm:ss", Locale.ENGLISH)
  };

  public static final int FEATURES = 10000;
  // 1997-01-15 00:01:00 GMT
  private static final long DATE_REFERENCE = 853286460;
  private static final long MONTH = 30 * 24 * 3600;
  private static final long WEEK = 7 * 24 * 3600;
  
  private final Random rand = RandomUtils.getRandom();  
  private final Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_46);
  private final FeatureVectorEncoder encoder = new StaticWordValueEncoder("body");
  private final FeatureVectorEncoder bias = new ConstantValueEncoder("Intercept");
  
  public FeatureVectorEncoder getEncoder() {
    return encoder;
  }
  
  public FeatureVectorEncoder getBias() {
    return bias;
  }
  
  public Random getRandom() {
    return rand;
  }

  public Vector encodeFeatureVector(File file, int actual, int leakType, Multiset<String> overallCounts)
    throws IOException {
    long date = (long) (1000 * (DATE_REFERENCE + actual * MONTH + 1 * WEEK * rand.nextDouble()));
    Multiset<String> words = ConcurrentHashMultiset.create();

    BufferedReader reader = Files.newReader(file, Charsets.UTF_8);
    try {
      String line = reader.readLine();
      Reader dateString = new StringReader(DATE_FORMATS[leakType % 3].format(new Date(date)));
      countWords(analyzer, words, dateString, overallCounts);
      while (line != null && !line.isEmpty()) {
        boolean countHeader = (
                line.startsWith("From:") || line.startsWith("Subject:")
                        || line.startsWith("Keywords:") || line.startsWith("Summary:")) && leakType < 6;
        do {
          Reader in = new StringReader(line);
          if (countHeader) {
            countWords(analyzer, words, in, overallCounts);
          }
          line = reader.readLine();
        } while (line != null && line.startsWith(" "));
      }
      if (leakType < 3) {
        countWords(analyzer, words, reader, overallCounts);
      }
    } finally {
      Closeables.close(reader, true);
    }

    Vector v = new RandomAccessSparseVector(FEATURES);
    bias.addToVector("", 1, v);
    for (String word : words.elementSet()) {
      encoder.addToVector(word, Math.log1p(words.count(word)), v);
    }

    return v;
  }

  public static void countWords(Analyzer analyzer,
                                 Collection<String> words,
                                 Reader in,
                                 Multiset<String> overallCounts) throws IOException {
    TokenStream ts = analyzer.tokenStream("text", in);
    ts.addAttribute(CharTermAttribute.class);
    ts.reset();
    while (ts.incrementToken()) {
      String s = ts.getAttribute(CharTermAttribute.class).toString();
      words.add(s);
    }
    overallCounts.addAll(words);
    ts.end();
    Closeables.close(ts, true);
  }
}
