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

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.mahout.classifier.NewsgroupHelper;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.Dictionary;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.common.collect.Ordering;

/**
 * Reads and trains an adaptive logistic regression model on the 20 newsgroups data.
 * The first command line argument gives the path of the directory holding the training
 * data.  The optional second argument, leakType, defines which classes of features to use.
 * Importantly, leakType controls whether a synthetic date is injected into the data as
 * a target leak and if so, how.
 * <p/>
 * The value of leakType % 3 determines whether the target leak is injected according to
 * the following table:
 * <p/>
 * <table>
 * <tr><td valign='top'>0</td><td>No leak injected</td></tr>
 * <tr><td valign='top'>1</td><td>Synthetic date injected in MMM-yyyy format. This will be a single token and
 * is a perfect target leak since each newsgroup is given a different month</td></tr>
 * <tr><td valign='top'>2</td><td>Synthetic date injected in dd-MMM-yyyy HH:mm:ss format.  The day varies
 * and thus there are more leak symbols that need to be learned.  Ultimately this is just
 * as big a leak as case 1.</td></tr>
 * </table>
 * <p/>
 * Leaktype also determines what other text will be indexed.  If leakType is greater
 * than or equal to 6, then neither headers nor text body will be used for features and the leak is the only
 * source of data.  If leakType is greater than or equal to 3, then subject words will be used as features.
 * If leakType is less than 3, then both subject and body text will be used as features.
 * <p/>
 * A leakType of 0 gives no leak and all textual features.
 * <p/>
 * See the following table for a summary of commonly used values for leakType
 * <p/>
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
public final class TrainNewsGroups {

  private TrainNewsGroups() {
  }

  public static void main(String[] args) throws IOException {
    File base = new File(args[0]);

    Multiset<String> overallCounts = HashMultiset.create();

    int leakType = 0;
    if (args.length > 1) {
      leakType = Integer.parseInt(args[1]);
    }

    Dictionary newsGroups = new Dictionary();

    NewsgroupHelper helper = new NewsgroupHelper();
    helper.getEncoder().setProbes(2);
    AdaptiveLogisticRegression learningAlgorithm =
        new AdaptiveLogisticRegression(20, NewsgroupHelper.FEATURES, new L1());
    learningAlgorithm.setInterval(800);
    learningAlgorithm.setAveragingWindow(500);

    List<File> files = Lists.newArrayList();
    for (File newsgroup : base.listFiles()) {
      if (newsgroup.isDirectory()) {
        newsGroups.intern(newsgroup.getName());
        files.addAll(Arrays.asList(newsgroup.listFiles()));
      }
    }
    Collections.shuffle(files);
    System.out.println(files.size() + " training files");
    SGDInfo info = new SGDInfo();

    int k = 0;


    for (File file : files) {
      String ng = file.getParentFile().getName();
      int actual = newsGroups.intern(ng);

      Vector v = helper.encodeFeatureVector(file, actual, leakType, overallCounts);
      learningAlgorithm.train(actual, v);

      k++;
      State<AdaptiveLogisticRegression.Wrapper, CrossFoldLearner> best = learningAlgorithm.getBest();

      SGDHelper.analyzeState(info, leakType, k, best);
    }
    learningAlgorithm.close();
    SGDHelper.dissect(leakType, newsGroups, learningAlgorithm, files, overallCounts);
    System.out.println("exiting main");

    ModelSerializer.writeBinary("/tmp/news-group.model",
            learningAlgorithm.getBest().getPayload().getLearner().getModels().get(0));

    List<Integer> counts = Lists.newArrayList();
    System.out.println("Word counts");
    for (String count : overallCounts.elementSet()) {
      counts.add(overallCounts.count(count));
    }
    Collections.sort(counts, Ordering.natural().reverse());
    k = 0;
    for (Integer count : counts) {
      System.out.println(k + "\t" + count);
      k++;
      if (k > 1000) {
        break;
      }
    }
  }


}
