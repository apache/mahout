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

package org.apache.mahout.utils.vectors.text;

import java.io.File;
import java.io.IOException;
import java.io.StringReader;
import java.net.URISyntaxException;
import java.util.*;

import junit.framework.TestCase;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.lang.mutable.MutableInt;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.SimpleAnalyzer;
import org.apache.lucene.analysis.Token;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.math.SparseVector;

/**
 * Test the dictionary Vector
 * 
 */
public class DictionaryVectorizerTest extends TestCase {
  
  public static final int AVG_DOCUMENT_LENGTH = 20;
  
  public static final int AVG_SENTENCE_LENGTH = 8;
  
  public static final int AVG_WORD_LENGTH = 6;
  
  public static final int NUM_DOCS = 100;
  
  public static final String CHARSET = "abcdef";
  
  public static final String DELIM = " .,?;:!\t\n\r";
  
  public static final String ERRORSET = "`1234567890"
                                        + "-=~@#$%^&*()_+[]{}'\"/<>|\\";
  
  private static Random random = new Random();
  
  private FileSystem fs;
  
  private static char getRandomDelimiter() {
    return DELIM.charAt(random.nextInt(DELIM.length()));
  }
  
  private static String getRandomDocument() {
    int length = (AVG_DOCUMENT_LENGTH >> 1)
                 + random.nextInt(AVG_DOCUMENT_LENGTH);
    StringBuilder sb = new StringBuilder(length * AVG_SENTENCE_LENGTH
                                         * AVG_WORD_LENGTH);
    for (int i = 0; i < length; i++) {
      sb.append(getRandomSentence());
    }
    return sb.toString();
  }
  
  private static String getRandomSentence() {
    int length = (AVG_SENTENCE_LENGTH >> 1)
                 + random.nextInt(AVG_SENTENCE_LENGTH);
    StringBuilder sb = new StringBuilder(length * AVG_WORD_LENGTH);
    for (int i = 0; i < length; i++) {
      sb.append(getRandomString()).append(' ');
    }
    sb.append(getRandomDelimiter());
    return sb.toString();
  }
  
  private static String getRandomString() {
    int length = (AVG_WORD_LENGTH >> 1) + random.nextInt(AVG_WORD_LENGTH);
    StringBuilder sb = new StringBuilder(length);
    for (int i = 0; i < length; i++) {
      sb.append(CHARSET.charAt(random.nextInt(CHARSET.length())));
    }
    if (random.nextInt(10) == 0) sb.append(ERRORSET.charAt(random
        .nextInt(ERRORSET.length())));
    return sb.toString();
  }
  
  private static void rmr(String path) throws Exception {
    File f = new File(path);
    if (f.exists()) {
      if (f.isDirectory()) {
        String[] contents = f.list();
        for (String content : contents) {
          rmr(f.toString() + File.separator + content);
        }
      }
      f.delete();
    }
  }
  
  public void setUp() throws Exception {
    rmr("output");
    rmr("testdata");
    Configuration conf = new Configuration();
    fs = FileSystem.get(conf);
  }
  
  public void testCreateTermFrequencyVectors() throws IOException,
                                              InterruptedException,
                                              ClassNotFoundException,
                                              URISyntaxException {
    Configuration conf = new Configuration();
    String pathString = "testdata/documents/docs.file";
    Path path = new Path(pathString);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path,
        Text.class, Text.class);
    
    for (int i = 0; i < NUM_DOCS; i++) {
      writer.append(new Text("Document::ID::" + i), new Text(
          getRandomDocument()));
    }
    writer.close();
    DictionaryVectorizer.createTermFrequencyVectors(pathString,
      "output/wordcount", new StandardAnalyzer(), 2, 100);
    
    
  }

  public void testPerf() throws Exception {
    Analyzer analyzer = new SimpleAnalyzer();
    String key = "key";
    String value = "";
    for(String doc : DOCS) value += doc + " ";
    Map<String, Integer> dictionary = new HashMap<String,Integer>();

    TokenStream ts = analyzer.tokenStream(key.toString(), new StringReader(value.toString()));

    Token token = new Token();
    int count = 0;
    while ((token = ts.next(token)) != null) {
      String tk = new String(token.termBuffer(), 0, token.termLength());
      if(dictionary.containsKey(tk)) continue;
      dictionary.put(tk, count++);
    }


    long vectorOnlyTotal = 0;
    long total = 0;

    Random rand = new Random(12345);
    String[] docs = generateRandomText(1000);

    for(int i=0; i<21000; i++) {

      long time = System.nanoTime();

      value = docs[rand.nextInt(docs.length)];
      ts = analyzer.tokenStream(key.toString(), new StringReader(value.toString()));

      SparseVector vector;
      Map<String,MutableInt> termFrequency = new HashMap<String,MutableInt>();

      token = new Token();
      ts.reset();
      while ((token = ts.next(token)) != null) {
        String tk = new String(token.termBuffer(), 0, token.termLength());
        if(dictionary.containsKey(tk) == false) continue;
        if (termFrequency.containsKey(tk) == false) {
          count += tk.length() + 1;
          termFrequency.put(tk, new MutableInt(0));
        }
        termFrequency.get(tk).increment();
      }

      vector =
          new SparseVector(key.toString(), Integer.MAX_VALUE, termFrequency.size());

      for (Map.Entry<String,MutableInt> pair : termFrequency.entrySet()) {
        String tk = pair.getKey();
        if (dictionary.containsKey(tk) == false) continue;
        vector.setQuick(dictionary.get(tk).intValue(), pair.getValue()
            .doubleValue());
      }
      total += (i<1000?0:1)*(System.nanoTime() - time);

      time = System.nanoTime();


      value = docs[rand.nextInt(docs.length)];
      ts = analyzer.tokenStream(key.toString(), new StringReader(value.toString()));
      
      vector =
          new SparseVector(key.toString(), Integer.MAX_VALUE, 10);

      token = new Token();
      ts.reset();
      while ((token = ts.next(token)) != null) {
        String tk = new String(token.termBuffer(), 0, token.termLength());
        if(dictionary.containsKey(tk) == false) continue;
        int tokenKey = dictionary.get(tk);
        vector.setQuick(tokenKey, vector.getQuick(tokenKey) + 1);
      }
      vectorOnlyTotal += (i<1000?0:1)*(System.nanoTime() - time);


    }

    System.out.println("With map: " + (total / 1e6) + "ms/KVect, with vector only: " + (vectorOnlyTotal/1e6) + "ms/KVect");

  }
  private static final String [] DOCS = {
        "The quick red fox jumped over the lazy brown dogs.",
        "Mary had a little lamb whose fleece was white as snow.",
        "Moby Dick is a story of a whale and a man obsessed.",
        "The robber wore a black fleece jacket and a baseball cap.",
        "The English Springer Spaniel is the best of all dogs."
    };

  public static String[] generateRandomText(int docs) throws Exception {
    String[] s = new String[docs];
    Random r = new Random(1234);
    for(int i=0; i<s.length; i++) {
      String str = DOCS[i % DOCS.length];
      String[] tokens = str.split(" ");
      String[] other = DOCS[r.nextInt(DOCS.length)].split(" ");
      List<String> l = new ArrayList<String>();
      for(String t : tokens) {
        l.add(r.nextBoolean() ? t : other[r.nextInt(other.length)]);
      }
      s[i] = StringUtils.join(l.toArray(new String[l.size()]), " ");
    }
    return s;
  }
  
}
