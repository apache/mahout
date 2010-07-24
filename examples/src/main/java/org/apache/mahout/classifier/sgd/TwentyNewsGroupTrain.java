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

package org.apache.mahout.classifier.sgd;

import com.google.common.base.Splitter;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multiset;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.*;

/**
 * Simple training program that reads newsgroup articles, one per file and trains an SGD model using
 * that data.
 */
public class TwentyNewsGroupTrain {
  private static final int FEATURES = 200000;
  private static final int PASSES = 1;
  private static Splitter onColon = Splitter.on(":").trimResults();


  public static void main(String[] args) throws IOException {
    File base = new File(args[0]);

    Map<String, Set<Integer>> traceDictionary = Maps.newTreeMap();
    RecordValueEncoder encoder = new StaticWordValueEncoder("body");
    RecordValueEncoder bias = new ConstantValueEncoder("Intercept");
    bias.setTraceDictionary(traceDictionary);
    bias.setTraceDictionary(traceDictionary);
    RecordValueEncoder lines = new ConstantValueEncoder("Lines");
    RecordValueEncoder logLines = new ConstantValueEncoder("LogLines");
    encoder.setProbes(2);
    encoder.setTraceDictionary(traceDictionary);

    OnlineLogisticRegression learningAlgorithm = new OnlineLogisticRegression(20, FEATURES, new L1())
            .alpha(1)
            .stepOffset(1000)
            .decayExponent(0.9)
            .lambda(0)
            .learningRate(10);

    Dictionary newsGroups = new Dictionary();

    List<File> files = Lists.newArrayList();
    for (File newsgroup : base.listFiles()) {
      newsGroups.intern(newsgroup.getName());
      files.addAll(Arrays.asList(newsgroup.listFiles()));
    }
    System.out.printf("%d files\n", files.size());

    Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_30);
    Random rand = new Random();
    double averageLL = 0;
    double averageCorrect = 0;
    double averageLineCount = 0;

    int k = 0;
    double step = 0;
    int[] bumps = new int[]{1, 2, 5};
    PrintWriter lineCounts = new PrintWriter(new File("lineCounts.tsv"));
    for (File file : permute(files, rand)) {
      BufferedReader reader = new BufferedReader(new FileReader(file));

      String ng = file.getParentFile().getName();
      int actual = newsGroups.intern(ng);

      Multiset<String> words = ConcurrentHashMultiset.create();

      double lineCount = averageLineCount;

      // read headers
      String line = reader.readLine();
      while (line != null && line.length() > 0) {
        if (line.startsWith("Lines:")) {
          String count = Lists.newArrayList(onColon.split(line)).get(1);
          try {
            lineCount = Integer.parseInt(count);
            averageLineCount = averageLineCount + (lineCount - averageLineCount) / Math.min(k + 1, 1000);
            lineCounts.printf("%s\t%.1f\n", ng, lineCount);
          } catch (NumberFormatException e) {
            // ignore bogus data, use average value
          }
        }

        boolean countHeader = (
                false
//                 ||line.startsWith("From:")
//                        ||line.startsWith("Subject:")
//                        ||line.startsWith("Keywords:")
                        ||line.startsWith("Summary:")
                );
        do {
          StringReader in = new StringReader(line);
          if (countHeader) {
            countWords(analyzer, words, in);
          }
          line = reader.readLine();
        } while (line.startsWith(" "));
      }

      // read body of document
      //      countWords(analyzer, words, reader);
      reader.close();

      // now encode words as vector
      Vector v = new RandomAccessSparseVector(FEATURES);

      // encode constant term
      bias.addToVector(null, 1, v);

      lines.addToVector(null, lineCount / 30, v);
      logLines.addToVector(null, Math.log(lineCount + 1), v);

      // and then all other words
      for (String word : words.elementSet()) {
        encoder.addToVector(word, Math.log(1 + words.count(word)), v);
      }

      double ll = learningAlgorithm.logLikelihood(actual, v);
      averageLL = (Math.min(k, 100) * averageLL + ll) / (Math.min(k, 100) + 1);
      Vector p = new DenseVector(20);
      learningAlgorithm.classifyFull(p, v);
      int estimated = p.maxValueIndex();
      boolean correct = estimated == actual;
      averageCorrect = (Math.min(k, 500) * averageCorrect + (correct ? 1 : 0)) / (Math.min(k, 500) + 1);
      learningAlgorithm.train(actual, v);

      k++;
      if (k % (bumps[(int) Math.floor(step) % bumps.length] * Math.pow(10, Math.floor(step / bumps.length))) == 0) {
        step += 0.25;
        if (estimated == -1) {
          System.out.printf("%d\n", estimated);
        }
        System.out.printf("%10d %10.3f %10.3f %10.2f %s %s\n",
                k, ll, averageLL, averageCorrect * 100, ng, newsGroups.values().get(estimated));

      }
      lineCounts.close();
    }

    learningAlgorithm.close();

    GsonBuilder gb = new GsonBuilder();
    gb.registerTypeAdapter(Matrix.class, new LogisticModelParameters.MatrixTypeAdapter());
    Gson gson = gb.setPrettyPrinting().create();

    Writer output = new FileWriter("model");

    Model x = new Model();
    x.lr = learningAlgorithm;
    x.targetCategories = newsGroups.values();
    gson.toJson(x, output);

    output.close();
  }

  private static void checkVector(Vector v) {
    Iterator<Vector.Element> i = v.iterateNonZero();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      if (Double.isInfinite(element.get()) || Double.isNaN(element.get())) {
        System.out.printf("Found invalid value at %d: %.0f\n", element.index(), element.get());
      }
    }
  }

  private static class Model {
    OnlineLogisticRegression lr;
    List<String> targetCategories;
  }

  private static void countWords(Analyzer analyzer, Multiset<String> words, Reader in) throws IOException {
    TokenStream ts = analyzer.tokenStream("body", in);
    TermAttribute termAtt = ts.addAttribute(TermAttribute.class);
    while (ts.incrementToken()) {
      char[] termBuffer = termAtt.termBuffer();
      int termLen = termAtt.termLength();
      words.add(new String(termBuffer, 0, termLen));
    }
  }

  private static <T> List<T> permute(Iterable<T> values, Random rand) {
    ArrayList<T> r = Lists.newArrayList();
    for (T value : values) {
      int i = rand.nextInt(r.size() + 1);
      if (i < r.size()) {
        T t = r.get(i);
        r.set(i, value);
        r.add(t);
      } else {
        r.add(value);
      }
    }
    return r;
  }
}
