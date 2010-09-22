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
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.UnaryFunction;
import org.apache.mahout.vectors.ConstantValueEncoder;
import org.apache.mahout.vectors.Dictionary;
import org.apache.mahout.vectors.FeatureVectorEncoder;
import org.apache.mahout.vectors.StaticWordValueEncoder;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Reads and trains an adaptive logistic regression model on the 20 newsgroups data.
 * The first command line argument gives the path of the directory holding the training
 * data.  The optional second argument, leakType, defines which classes of features to use.
 * Importantly, leakType controls whether a synthetic date is injected into the data as
 * a target leak and if so, how.
 * <p>
 * The value of leakType % 3 determines whether the target leak is injected according to
 * the following table:
 * <p>
 * <table>
 * <tr><td valign='top'>0</td><td>No leak injected</td></tr>
 * <tr><td valign='top'>1</td><td>Synthetic date injected in MMM-yyyy format. This will be a single token and
 * is a perfect target leak since each newsgroup is given a different month</td></tr>
 * <tr><td valign='top'>2</td><td>Synthetic date injected in dd-MMM-yyyy HH:mm:ss format.  The day varies
 * and thus there are more leak symbols that need to be learned.  Ultimately this is just
 * as big a leak as case 1.</td></tr>
 * </table>
 * <p>
 * Leaktype also determines what other text will be indexed.  If leakType is greater
 * than or equal to 6, then neither headers nor text body will be used for features and the leak is the only
 * source of data.  If leakType is greater than or equal to 3, then subject words will be used as features.
 * If leakType is less than 3, then both subject and body text will be used as features.
 * <p>
 * A leakType of 0 gives no leak and all textual features.
 * <p>
 * See the following table for a summary of commonly used values for leakType
 * <p>
 * <table>
 * <tr><td><b>leakType</b></td><td><b>Leak?</b></td><td><b>Subject?</b></td><td><b>Body?</b></td></tr>
 * <tr><td colspan=4><hr></td></tr>
 * <tr><td>0</td><td>no</td><td>yes</td><td>yes</td></tr>
 * <tr><td>1</td><td>mmm-yyyy</td><td>yes</td><td>yes</td></tr>
 * <tr><td>2</td><td>dd-mmm-yyyy</td><td>yes</td><td>yes</td></tr>
 * <tr><td colspan=4><hr></td></tr>
 * <tr><td>3</td><td>no</td><td>yes</td><td>no</td></tr>
 * <tr><td>4</td><td>mmm-yyyy</td><td>yes</td><td>no</td></tr>
 * <tr><td>5</td><td>dd-mmm-yyyy</td><td>yes</td><td>no</td></tr>
 * <tr><td colspan=4><hr></td></tr>
 * <tr><td>6</td><td>no</td><td>no</td><td>no</td></tr>
 * <tr><td>7</td><td>mmm-yyyy</td><td>no</td><td>no</td></tr>
 * <tr><td>8</td><td>dd-mmm-yyyy</td><td>no</td><td>no</td></tr>
 * <tr><td colspan=4><hr></td></tr>
 * </table>
 */
public class TrainNewsGroups {
  private static final int FEATURES = 10000;
  // 1997-01-15 00:01:00 GMT
  private static final long DATE_REFERENCE = 853286460;
  private static final long MONTH = 30 * 24 * 3600;
  private static final long WEEK = 7 * 24 * 3600;

  private static final Random rand = new Random();

  private static final Splitter ON_COLON = Splitter.on(":");

  private static final String[] leakLabels = {"none", "month-year", "day-month-year"};
  private static final SimpleDateFormat[] df = new SimpleDateFormat[]{
    new SimpleDateFormat(""),
    new SimpleDateFormat("MMM-yyyy"),
    new SimpleDateFormat("dd-MMM-yyyy HH:mm:ss")
  };

  private static final Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_30);
  private static final FeatureVectorEncoder encoder = new StaticWordValueEncoder("body");
  private static final FeatureVectorEncoder bias = new ConstantValueEncoder("Intercept");

  public static void main(String[] args) throws IOException {
    File base = new File(args[0]);

    int leakType = 0;
    if (args.length > 1) {
      leakType = Integer.parseInt(args[1]);
    }

    Dictionary newsGroups = new Dictionary();

    encoder.setProbes(2);
    AdaptiveLogisticRegression learningAlgorithm = new AdaptiveLogisticRegression(20, FEATURES, new L1());
    learningAlgorithm.setInterval(200);
    learningAlgorithm.setAveragingWindow(500);

    List<File> files = Lists.newArrayList();
    for (File newsgroup : base.listFiles()) {
      newsGroups.intern(newsgroup.getName());
      files.addAll(Arrays.asList(newsgroup.listFiles()));
    }
    System.out.printf("%d training files\n", files.size());

    double averageLL = 0;
    double averageCorrect = 0;

    int k = 0;
    double step = 0;
    int[] bumps = new int[]{1, 2, 5};
    for (File file : permute(files, rand).subList(0, 5000)) {
      String ng = file.getParentFile().getName();
      int actual = newsGroups.intern(ng);

      Vector v = encodeFeatureVector(file, actual, leakType);
      learningAlgorithm.train(actual, v);

      k++;

      int bump = bumps[(int) Math.floor(step) % bumps.length];
      int scale = (int) Math.pow(10, Math.floor(step / bumps.length));
      State<AdaptiveLogisticRegression.Wrapper> best = learningAlgorithm.getBest();
      double maxBeta;
      double nonZeros;
      double positive;
      double norm;

      double lambda = 0;
      double mu = 0;
      
      if (best != null) {
        CrossFoldLearner state = best.getPayload().getLearner();
        averageCorrect = state.percentCorrect();
        averageLL = state.logLikelihood();

        maxBeta = state.getModels().get(0).getBeta().aggregate(Functions.MAX, Functions.IDENTITY);
        nonZeros = state.getModels().get(0).getBeta().aggregate(Functions.PLUS, new UnaryFunction() {
          @Override
          public double apply(double v) {
            return Math.abs(v) > 1e-6 ? 1 : 0;
          }
        });
        positive = state.getModels().get(0).getBeta().aggregate(Functions.PLUS, new UnaryFunction() {
          @Override
          public double apply(double v) {
            return v > 0 ? 1 : 0;
          }
        });
        norm = state.getModels().get(0).getBeta().aggregate(Functions.PLUS, Functions.ABS);

        lambda = learningAlgorithm.getBest().getMappedParams()[0];
        mu = learningAlgorithm.getBest().getMappedParams()[1];
      } else {
        maxBeta = 0;
        nonZeros = 0;
        positive = 0;
        norm = 0;
      }
      if (k % (bump * scale) == 0) {
        step += 0.25;
        System.out.printf("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t", maxBeta, nonZeros, positive, norm, lambda, mu);
        System.out.printf("%d\t%.3f\t%.2f\t%s\t%s\n",
          k, averageLL, averageCorrect * 100, ng, leakLabels[leakType % 3]);
      }
    }
    learningAlgorithm.close();

    dissect(leakType, newsGroups, learningAlgorithm, files);
    System.out.println("exiting main");
  }

  private static void dissect(int leakType, Dictionary newsGroups, AdaptiveLogisticRegression learningAlgorithm, List<File> files) throws IOException {
    Map<String, Set<Integer>> traceDictionary = Maps.newTreeMap();
    System.out.printf("starting dissection\n");
    ModelDissector md = new ModelDissector(learningAlgorithm.getBest().getPayload().getLearner().numCategories());

    encoder.setTraceDictionary(traceDictionary);
    bias.setTraceDictionary(traceDictionary);
    int k = 0;
    for (File file : permute(files, rand).subList(0, 500)) {
      String ng = file.getParentFile().getName();
      int actual = newsGroups.intern(ng);

      traceDictionary.clear();
      Vector v = encodeFeatureVector(file, actual, leakType);
      md.update(v, traceDictionary, learningAlgorithm.getBest().getPayload().getLearner());
      if (k % 50 == 0) {
        System.out.printf("%d\t%d\n", k, traceDictionary.size());
      }
    }

    List<String> ngNames = Lists.newArrayList(newsGroups.values());
    List<ModelDissector.Weight> weights = md.summary(100);
    for (ModelDissector.Weight w : weights) {
      System.out.printf("%s\t%.1f\t%s\n", w.getFeature(), w.getWeight(), ngNames.get(w.getMaxImpact() + 1));
    }
  }

  private static Vector encodeFeatureVector(File file, int actual, int leakType) throws IOException {
    long date = (long) (1000 * (DATE_REFERENCE + actual * MONTH + 1 * WEEK * rand.nextDouble()));
    Multiset<String> words = ConcurrentHashMultiset.create();

    BufferedReader reader = new BufferedReader(new FileReader(file));
    String line = reader.readLine();
    Reader dateString = new StringReader(df[leakType % 3].format(new Date(date)));
    countWords(analyzer, words, dateString);
    while (line != null && line.length() > 0) {
      boolean countHeader = (
        line.startsWith("From:") || line.startsWith("Subject:") ||
          line.startsWith("Keywords:") || line.startsWith("Summary:")) && (leakType < 6);
      do {
        StringReader in = new StringReader(line);
        if (countHeader) {
          countWords(analyzer, words, in);
        }
        line = reader.readLine();
      } while (line.startsWith(" "));
    }
    if (leakType < 3) {
      countWords(analyzer, words, reader);
    }
    reader.close();

    Vector v = new RandomAccessSparseVector(FEATURES);
    bias.addToVector("", 1, v);
    for (String word : words.elementSet()) {
      encoder.addToVector(word, Math.log(1 + words.count(word)), v);
    }

    return v;
  }

  private static void countWords(Analyzer analyzer, Multiset<String> words, Reader in) throws IOException {
    TokenStream ts = analyzer.tokenStream("text", in);
    ts.addAttribute(TermAttribute.class);
    while (ts.incrementToken()) {
      String s = ts.getAttribute(TermAttribute.class).term();
      words.add(s);
    }
  }

  private static List<File> permute(List<File> files, Random rand) {
    List<File> r = Lists.newArrayList();
    for (File file : files) {
      int i = rand.nextInt(r.size() + 1);
      if (i == r.size()) {
        r.add(file);
      } else {
        r.add(r.get(i));
        r.set(i, file);
      }
    }
    return r;
  }

}