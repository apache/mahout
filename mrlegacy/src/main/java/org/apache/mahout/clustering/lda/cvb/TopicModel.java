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
package org.apache.mahout.clustering.lda.cvb;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DistributedRowMatrixWriter;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.stats.Sampler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Thin wrapper around a {@link Matrix} of counts of occurrences of (topic, term) pairs.  Dividing
 * {code topicTermCount.viewRow(topic).get(term)} by the sum over the values for all terms in that
 * row yields p(term | topic).  Instead dividing it by all topic columns for that term yields
 * p(topic | term).
 *
 * Multithreading is enabled for the {@code update(Matrix)} method: this method is async, and
 * merely submits the matrix to a work queue.  When all work has been submitted,
 * {@code awaitTermination()} should be called, which will block until updates have been
 * accumulated.
 */
public class TopicModel implements Configurable, Iterable<MatrixSlice> {
  
  private static final Logger log = LoggerFactory.getLogger(TopicModel.class);
  
  private final String[] dictionary;
  private final Matrix topicTermCounts;
  private final Vector topicSums;
  private final int numTopics;
  private final int numTerms;
  private final double eta;
  private final double alpha;

  private Configuration conf;

  private final Sampler sampler;
  private final int numThreads;
  private ThreadPoolExecutor threadPool;
  private Updater[] updaters;

  public int getNumTerms() {
    return numTerms;
  }

  public int getNumTopics() {
    return numTopics;
  }

  public TopicModel(int numTopics, int numTerms, double eta, double alpha, String[] dictionary,
      double modelWeight) {
    this(numTopics, numTerms, eta, alpha, null, dictionary, 1, modelWeight);
  }

  public TopicModel(Configuration conf, double eta, double alpha,
      String[] dictionary, int numThreads, double modelWeight, Path... modelpath) throws IOException {
    this(loadModel(conf, modelpath), eta, alpha, dictionary, numThreads, modelWeight);
  }

  public TopicModel(int numTopics, int numTerms, double eta, double alpha, String[] dictionary,
      int numThreads, double modelWeight) {
    this(new DenseMatrix(numTopics, numTerms), new DenseVector(numTopics), eta, alpha, dictionary,
        numThreads, modelWeight);
  }

  public TopicModel(int numTopics, int numTerms, double eta, double alpha, Random random,
      String[] dictionary, int numThreads, double modelWeight) {
    this(randomMatrix(numTopics, numTerms, random), eta, alpha, dictionary, numThreads, modelWeight);
  }

  private TopicModel(Pair<Matrix, Vector> model, double eta, double alpha, String[] dict,
      int numThreads, double modelWeight) {
    this(model.getFirst(), model.getSecond(), eta, alpha, dict, numThreads, modelWeight);
  }

  public TopicModel(Matrix topicTermCounts, Vector topicSums, double eta, double alpha,
    String[] dictionary, double modelWeight) {
    this(topicTermCounts, topicSums, eta, alpha, dictionary, 1, modelWeight);
  }

  public TopicModel(Matrix topicTermCounts, double eta, double alpha, String[] dictionary,
      int numThreads, double modelWeight) {
    this(topicTermCounts, viewRowSums(topicTermCounts),
        eta, alpha, dictionary, numThreads, modelWeight);
  }

  public TopicModel(Matrix topicTermCounts, Vector topicSums, double eta, double alpha,
    String[] dictionary, int numThreads, double modelWeight) {
    this.dictionary = dictionary;
    this.topicTermCounts = topicTermCounts;
    this.topicSums = topicSums;
    this.numTopics = topicSums.size();
    this.numTerms = topicTermCounts.numCols();
    this.eta = eta;
    this.alpha = alpha;
    this.sampler = new Sampler(RandomUtils.getRandom());
    this.numThreads = numThreads;
    if (modelWeight != 1) {
      topicSums.assign(Functions.mult(modelWeight));
      for (int x = 0; x < numTopics; x++) {
        topicTermCounts.viewRow(x).assign(Functions.mult(modelWeight));
      }
    }
    initializeThreadPool();
  }

  private static Vector viewRowSums(Matrix m) {
    Vector v = new DenseVector(m.numRows());
    for (MatrixSlice slice : m) {
      v.set(slice.index(), slice.vector().norm(1));
    }
    return v;
  }

  private synchronized void initializeThreadPool() {
    if (threadPool != null) {
      threadPool.shutdown();
      try {
        threadPool.awaitTermination(100, TimeUnit.SECONDS);
      } catch (InterruptedException e) {
        log.error("Could not terminate all threads for TopicModel in time.", e);
      }
    }
    threadPool = new ThreadPoolExecutor(numThreads, numThreads, 0, TimeUnit.SECONDS,
                                                           new ArrayBlockingQueue<Runnable>(numThreads * 10));
    threadPool.allowCoreThreadTimeOut(false);
    updaters = new Updater[numThreads];
    for (int i = 0; i < numThreads; i++) {
      updaters[i] = new Updater();
      threadPool.submit(updaters[i]);
    }
  }

  Matrix topicTermCounts() {
    return topicTermCounts;
  }

  @Override
  public Iterator<MatrixSlice> iterator() {
    return topicTermCounts.iterateAll();
  }

  public Vector topicSums() {
    return topicSums;
  }

  private static Pair<Matrix,Vector> randomMatrix(int numTopics, int numTerms, Random random) {
    Matrix topicTermCounts = new DenseMatrix(numTopics, numTerms);
    Vector topicSums = new DenseVector(numTopics);
    if (random != null) {
      for (int x = 0; x < numTopics; x++) {
        for (int term = 0; term < numTerms; term++) {
          topicTermCounts.viewRow(x).set(term, random.nextDouble());
        }
      }
    }
    for (int x = 0; x < numTopics; x++) {
      topicSums.set(x, random == null ? 1.0 : topicTermCounts.viewRow(x).norm(1));
    }
    return Pair.of(topicTermCounts, topicSums);
  }

  public static Pair<Matrix, Vector> loadModel(Configuration conf, Path... modelPaths)
    throws IOException {
    int numTopics = -1;
    int numTerms = -1;
    List<Pair<Integer, Vector>> rows = Lists.newArrayList();
    for (Path modelPath : modelPaths) {
      for (Pair<IntWritable, VectorWritable> row
          : new SequenceFileIterable<IntWritable, VectorWritable>(modelPath, true, conf)) {
        rows.add(Pair.of(row.getFirst().get(), row.getSecond().get()));
        numTopics = Math.max(numTopics, row.getFirst().get());
        if (numTerms < 0) {
          numTerms = row.getSecond().get().size();
        }
      }
    }
    if (rows.isEmpty()) {
      throw new IOException(Arrays.toString(modelPaths) + " have no vectors in it");
    }
    numTopics++;
    Matrix model = new DenseMatrix(numTopics, numTerms);
    Vector topicSums = new DenseVector(numTopics);
    for (Pair<Integer, Vector> pair : rows) {
      model.viewRow(pair.getFirst()).assign(pair.getSecond());
      topicSums.set(pair.getFirst(), pair.getSecond().norm(1));
    }
    return Pair.of(model, topicSums);
  }

  // NOTE: this is purely for debug purposes.  It is not performant to "toString()" a real model
  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    for (int x = 0; x < numTopics; x++) {
      String v = dictionary != null
          ? vectorToSortedString(topicTermCounts.viewRow(x).normalize(1), dictionary)
          : topicTermCounts.viewRow(x).asFormatString();
      buf.append(v).append('\n');
    }
    return buf.toString();
  }

  public int sampleTerm(Vector topicDistribution) {
    return sampler.sample(topicTermCounts.viewRow(sampler.sample(topicDistribution)));
  }

  public int sampleTerm(int topic) {
    return sampler.sample(topicTermCounts.viewRow(topic));
  }

  public synchronized void reset() {
    for (int x = 0; x < numTopics; x++) {
      topicTermCounts.assignRow(x, new SequentialAccessSparseVector(numTerms));
    }
    topicSums.assign(1.0);
    if (threadPool.isTerminated()) {
      initializeThreadPool();
    }
  }

  public synchronized void stop() {
    for (Updater updater : updaters) {
      updater.shutdown();
    }
    threadPool.shutdown();
    try {
      if (!threadPool.awaitTermination(60, TimeUnit.SECONDS)) {
        log.warn("Threadpool timed out on await termination - jobs still running!");
      }
    } catch (InterruptedException e) {
      log.error("Interrupted shutting down!", e);
    }
  }

  public void renormalize() {
    for (int x = 0; x < numTopics; x++) {
      topicTermCounts.assignRow(x, topicTermCounts.viewRow(x).normalize(1));
      topicSums.assign(1.0);
    }
  }

  public void trainDocTopicModel(Vector original, Vector topics, Matrix docTopicModel) {
    // first calculate p(topic|term,document) for all terms in original, and all topics,
    // using p(term|topic) and p(topic|doc)
    pTopicGivenTerm(original, topics, docTopicModel);
    normalizeByTopic(docTopicModel);
    // now multiply, term-by-term, by the document, to get the weighted distribution of
    // term-topic pairs from this document.
    for (Element e : original.nonZeroes()) {
      for (int x = 0; x < numTopics; x++) {
        Vector docTopicModelRow = docTopicModel.viewRow(x);
        docTopicModelRow.setQuick(e.index(), docTopicModelRow.getQuick(e.index()) * e.get());
      }
    }
    // now recalculate \(p(topic|doc)\) by summing contributions from all of pTopicGivenTerm
    topics.assign(0.0);
    for (int x = 0; x < numTopics; x++) {
      topics.set(x, docTopicModel.viewRow(x).norm(1));
    }
    // now renormalize so that \(sum_x(p(x|doc))\) = 1
    topics.assign(Functions.mult(1 / topics.norm(1)));
  }

  public Vector infer(Vector original, Vector docTopics) {
    Vector pTerm = original.like();
    for (Element e : original.nonZeroes()) {
      int term = e.index();
      // p(a) = sum_x (p(a|x) * p(x|i))
      double pA = 0;
      for (int x = 0; x < numTopics; x++) {
        pA += (topicTermCounts.viewRow(x).get(term) / topicSums.get(x)) * docTopics.get(x);
      }
      pTerm.set(term, pA);
    }
    return pTerm;
  }

  public void update(Matrix docTopicCounts) {
    for (int x = 0; x < numTopics; x++) {
      updaters[x % updaters.length].update(x, docTopicCounts.viewRow(x));
    }
  }

  public void updateTopic(int topic, Vector docTopicCounts) {
    topicTermCounts.viewRow(topic).assign(docTopicCounts, Functions.PLUS);
    topicSums.set(topic, topicSums.get(topic) + docTopicCounts.norm(1));
  }

  public void update(int termId, Vector topicCounts) {
    for (int x = 0; x < numTopics; x++) {
      Vector v = topicTermCounts.viewRow(x);
      v.set(termId, v.get(termId) + topicCounts.get(x));
    }
    topicSums.assign(topicCounts, Functions.PLUS);
  }

  public void persist(Path outputDir, boolean overwrite) throws IOException {
    FileSystem fs = outputDir.getFileSystem(conf);
    if (overwrite) {
      fs.delete(outputDir, true); // CHECK second arg
    }
    DistributedRowMatrixWriter.write(outputDir, conf, topicTermCounts);
  }

  /**
   * Computes {@code \(p(topic x | term a, document i)\)} distributions given input document {@code i}.
   * {@code \(pTGT[x][a]\)} is the (un-normalized) {@code \(p(x|a,i)\)}, or if docTopics is {@code null},
   * {@code \(p(a|x)\)} (also un-normalized).
   *
   * @param document doc-term vector encoding {@code \(w(term a|document i)\)}.
   * @param docTopics {@code docTopics[x]} is the overall weight of topic {@code x} in given
   *          document. If {@code null}, a topic weight of {@code 1.0} is used for all topics.
   * @param termTopicDist storage for output {@code \(p(x|a,i)\)} distributions.
   */
  private void pTopicGivenTerm(Vector document, Vector docTopics, Matrix termTopicDist) {
    // for each topic x
    for (int x = 0; x < numTopics; x++) {
      // get p(topic x | document i), or 1.0 if docTopics is null
      double topicWeight = docTopics == null ? 1.0 : docTopics.get(x);
      // get w(term a | topic x)
      Vector topicTermRow = topicTermCounts.viewRow(x);
      // get \sum_a w(term a | topic x)
      double topicSum = topicSums.get(x);
      // get p(topic x | term a) distribution to update
      Vector termTopicRow = termTopicDist.viewRow(x);

      // for each term a in document i with non-zero weight
      for (Element e : document.nonZeroes()) {
        int termIndex = e.index();

        // calc un-normalized p(topic x | term a, document i)
        double termTopicLikelihood = (topicTermRow.get(termIndex) + eta) * (topicWeight + alpha)
            / (topicSum + eta * numTerms);
        termTopicRow.set(termIndex, termTopicLikelihood);
      }
    }
  }

  /**
   * \(sum_x sum_a (c_ai * log(p(x|i) * p(a|x)))\)
   */
  public double perplexity(Vector document, Vector docTopics) {
    double perplexity = 0;
    double norm = docTopics.norm(1) + (docTopics.size() * alpha);
    for (Element e : document.nonZeroes()) {
      int term = e.index();
      double prob = 0;
      for (int x = 0; x < numTopics; x++) {
        double d = (docTopics.get(x) + alpha) / norm;
        double p = d * (topicTermCounts.viewRow(x).get(term) + eta)
                   / (topicSums.get(x) + eta * numTerms);
        prob += p;
      }
      perplexity += e.get() * Math.log(prob);
    }
    return -perplexity;
  }

  private void normalizeByTopic(Matrix perTopicSparseDistributions) {
    // then make sure that each of these is properly normalized by topic: sum_x(p(x|t,d)) = 1
    for (Element e : perTopicSparseDistributions.viewRow(0).nonZeroes()) {
      int a = e.index();
      double sum = 0;
      for (int x = 0; x < numTopics; x++) {
        sum += perTopicSparseDistributions.viewRow(x).get(a);
      }
      for (int x = 0; x < numTopics; x++) {
        perTopicSparseDistributions.viewRow(x).set(a,
            perTopicSparseDistributions.viewRow(x).get(a) / sum);
      }
    }
  }

  public static String vectorToSortedString(Vector vector, String[] dictionary) {
    List<Pair<String,Double>> vectorValues = Lists.newArrayListWithCapacity(vector.getNumNondefaultElements());
    for (Element e : vector.nonZeroes()) {
      vectorValues.add(Pair.of(dictionary != null ? dictionary[e.index()] : String.valueOf(e.index()),
                               e.get()));
    }
    Collections.sort(vectorValues, new Comparator<Pair<String, Double>>() {
      @Override public int compare(Pair<String, Double> x, Pair<String, Double> y) {
        return y.getSecond().compareTo(x.getSecond());
      }
    });
    Iterator<Pair<String,Double>> listIt = vectorValues.iterator();
    StringBuilder bldr = new StringBuilder(2048);
    bldr.append('{');
    int i = 0;
    while (listIt.hasNext() && i < 25) {
      i++;
      Pair<String,Double> p = listIt.next();
      bldr.append(p.getFirst());
      bldr.append(':');
      bldr.append(p.getSecond());
      bldr.append(',');
    }
    if (bldr.length() > 1) {
      bldr.setCharAt(bldr.length() - 1, '}');
    }
    return bldr.toString();
  }

  @Override
  public void setConf(Configuration configuration) {
    this.conf = configuration;
  }

  @Override
  public Configuration getConf() {
    return conf;
  }

  private final class Updater implements Runnable {
    private final ArrayBlockingQueue<Pair<Integer, Vector>> queue =
        new ArrayBlockingQueue<Pair<Integer, Vector>>(100);
    private boolean shutdown = false;
    private boolean shutdownComplete = false;

    public void shutdown() {
      try {
        synchronized (this) {
          while (!shutdownComplete) {
            shutdown = true;
            wait(10000L); // Arbitrarily, wait 10 seconds rather than forever for this
          }
        }
      } catch (InterruptedException e) {
        log.warn("Interrupted waiting to shutdown() : ", e);
      }
    }

    public boolean update(int topic, Vector v) {
      if (shutdown) { // maybe don't do this?
        throw new IllegalStateException("In SHUTDOWN state: cannot submit tasks");
      }
      while (true) { // keep trying if interrupted
        try {
          // start async operation by submitting to the queue
          queue.put(Pair.of(topic, v));
          // return once you got access to the queue
          return true;
        } catch (InterruptedException e) {
          log.warn("Interrupted trying to queue update:", e);
        }
      }
    }

    @Override
    public void run() {
      while (!shutdown) {
        try {
          Pair<Integer, Vector> pair = queue.poll(1, TimeUnit.SECONDS);
          if (pair != null) {
            updateTopic(pair.getFirst(), pair.getSecond());
          }
        } catch (InterruptedException e) {
          log.warn("Interrupted waiting to poll for update", e);
        }
      }
      // in shutdown mode, finish remaining tasks!
      for (Pair<Integer, Vector> pair : queue) {
        updateTopic(pair.getFirst(), pair.getSecond());
      }
      synchronized (this) {
        shutdownComplete = true;
        notifyAll();
      }
    }
  }

}
