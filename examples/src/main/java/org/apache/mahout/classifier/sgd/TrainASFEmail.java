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

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import com.google.common.collect.Ordering;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;

import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterator;
import org.apache.mahout.ep.State;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.encoders.Dictionary;

import java.io.File;
import java.util.Collections;
import java.util.List;

public final class TrainASFEmail extends AbstractJob {

  private TrainASFEmail() {
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption("categories", "nc", "The number of categories to train on", true);
    addOption("cardinality", "c", "The size of the vectors to use", "100000");
    addOption("threads", "t", "The number of threads to use in the learner", "20");
    addOption("poolSize", "p", "The number of CrossFoldLearners to use in the AdaptiveLogisticRegression. "
                               + "Higher values require more memory.", "5");
    if (parseArguments(args) == null) {
      return -1;
    }

    File base = new File(getInputPath().toString());

    Multiset<String> overallCounts = HashMultiset.create();
    File output = new File(getOutputPath().toString());
    output.mkdirs();
    int numCats = Integer.parseInt(getOption("categories"));
    int cardinality = Integer.parseInt(getOption("cardinality", "100000"));
    int threadCount = Integer.parseInt(getOption("threads", "20"));
    int poolSize = Integer.parseInt(getOption("poolSize", "5"));
    Dictionary asfDictionary = new Dictionary();
    AdaptiveLogisticRegression learningAlgorithm =
        new AdaptiveLogisticRegression(numCats, cardinality, new L1(), threadCount, poolSize);
    learningAlgorithm.setInterval(800);
    learningAlgorithm.setAveragingWindow(500);

    //We ran seq2encoded and split input already, so let's just build up the dictionary
    Configuration conf = new Configuration();
    PathFilter trainFilter = new PathFilter() {
      @Override
      public boolean accept(Path path) {
        return path.getName().contains("training");
      }
    };
    SequenceFileDirIterator<Text, VectorWritable> iter =
        new SequenceFileDirIterator<Text, VectorWritable>(new Path(base.toString()),
                                                          PathType.LIST,
                                                          trainFilter,
                                                          null,
                                                          true,
                                                          conf);
    long numItems = 0;
    while (iter.hasNext()) {
      Pair<Text, VectorWritable> next = iter.next();
      asfDictionary.intern(next.getFirst().toString());
      numItems++;
    }

    System.out.println(numItems + " training files");


    SGDInfo info = new SGDInfo();

    iter = new SequenceFileDirIterator<Text, VectorWritable>(new Path(base.toString()), PathType.LIST, trainFilter,
            null, true, conf);
    int k = 0;
    while (iter.hasNext()) {
      Pair<Text, VectorWritable> next = iter.next();
      String ng = next.getFirst().toString();
      int actual = asfDictionary.intern(ng);
      //we already have encoded
      learningAlgorithm.train(actual, next.getSecond().get());
      k++;
      State<AdaptiveLogisticRegression.Wrapper, CrossFoldLearner> best = learningAlgorithm.getBest();

      SGDHelper.analyzeState(info, 0, k, best);
    }
    learningAlgorithm.close();
    //TODO: how to dissection since we aren't processing the files here
    //SGDHelper.dissect(leakType, asfDictionary, learningAlgorithm, files, overallCounts);
    System.out.println("exiting main, writing model to " + output);

    ModelSerializer.writeBinary(output + "/asf.model",
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
    return 0;
  }

  public static void main(String[] args) throws Exception {
    TrainASFEmail trainer = new TrainASFEmail();
    trainer.run(args);
  }
}
