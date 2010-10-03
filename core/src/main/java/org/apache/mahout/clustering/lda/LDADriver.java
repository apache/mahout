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

package org.apache.mahout.clustering.lda;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.DenseMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * Estimates an LDA model from a corpus of documents, which are SparseVectors of word counts. At each phase,
 * it outputs a matrix of log probabilities of each topic.
 */
public final class LDADriver extends AbstractJob {

  private static final String TOPIC_SMOOTHING_OPTION = "topicSmoothing";

  private static final String NUM_WORDS_OPTION = "numWords";

  private static final String NUM_TOPICS_OPTION = "numTopics";

  static final String STATE_IN_KEY = "org.apache.mahout.clustering.lda.stateIn";

  static final String NUM_TOPICS_KEY = "org.apache.mahout.clustering.lda.numTopics";

  static final String NUM_WORDS_KEY = "org.apache.mahout.clustering.lda.numWords";

  static final String TOPIC_SMOOTHING_KEY = "org.apache.mahout.clustering.lda.topicSmoothing";

  static final int LOG_LIKELIHOOD_KEY = -2;

  static final int TOPIC_SUM_KEY = -1;

  static final double OVERALL_CONVERGENCE = 1.0E-5;

  private static final Logger log = LoggerFactory.getLogger(LDADriver.class);

  private LDADriver() {
  }

  public static void main(String[] args) throws Exception {
    new LDADriver().run(args);
  }

  static LDAState createState(Configuration job) throws IOException {
    String statePath = job.get(STATE_IN_KEY);
    int numTopics = Integer.parseInt(job.get(NUM_TOPICS_KEY));
    int numWords = Integer.parseInt(job.get(NUM_WORDS_KEY));
    double topicSmoothing = Double.parseDouble(job.get(TOPIC_SMOOTHING_KEY));

    Path dir = new Path(statePath);
    FileSystem fs = dir.getFileSystem(job);

    DenseMatrix pWgT = new DenseMatrix(numTopics, numWords);
    double[] logTotals = new double[numTopics];
    double ll = 0.0;

    IntPairWritable key = new IntPairWritable();
    DoubleWritable value = new DoubleWritable();
    for (FileStatus status : fs.globStatus(new Path(dir, "part-*"))) {
      Path path = status.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
      while (reader.next(key, value)) {
        int topic = key.getFirst();
        int word = key.getSecond();
        if (word == TOPIC_SUM_KEY) {
          logTotals[topic] = value.get();
          Preconditions.checkArgument(!Double.isInfinite(value.get()));
        } else if (topic == LOG_LIKELIHOOD_KEY) {
          ll = value.get();
        } else {
          Preconditions.checkArgument(topic >= 0, "topic should be non-negative, not %d", topic);
          Preconditions.checkArgument(word >= 0, "word should be non-negative not %d", word);
          Preconditions.checkArgument(pWgT.getQuick(topic, word) == 0.0);

          pWgT.setQuick(topic, word, value.get());
          Preconditions.checkArgument(!Double.isInfinite(pWgT.getQuick(topic, word)));
        }
      }
      reader.close();
    }

    return new LDAState(numTopics, numWords, topicSmoothing, pWgT, logTotals, ll);
  }

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(NUM_TOPICS_OPTION, "k", "The total number of topics in the corpus", true);
    addOption(NUM_WORDS_OPTION,
              "v",
              "The total number of words in the corpus (can be approximate, needs to exceed the actual value)");
    addOption(TOPIC_SMOOTHING_OPTION, "a", "Topic smoothing parameter. Default is 50/numTopics.", "-1.0");
    addOption(DefaultOptionCreator.maxIterationsOption().withRequired(false).create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.overwriteOutput(output);
    }
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    int numTopics = Integer.parseInt(getOption(NUM_TOPICS_OPTION));
    int numWords = Integer.parseInt(getOption(NUM_WORDS_OPTION));
    double topicSmoothing = Double.parseDouble(getOption(TOPIC_SMOOTHING_OPTION));
    if (topicSmoothing < 1) {
      topicSmoothing = 50.0 / numTopics;
    }

    run(getConf(), input, output, numTopics, numWords, topicSmoothing, maxIterations);

    return 0;
  }

  private static void run(Configuration conf,
                          Path input,
                          Path output,
                          int numTopics,
                          int numWords,
                          double topicSmoothing,
                          int maxIterations)
    throws IOException, InterruptedException, ClassNotFoundException {
    Path stateIn = new Path(output, "state-0");
    writeInitialState(stateIn, numTopics, numWords);
    double oldLL = Double.NEGATIVE_INFINITY;
    boolean converged = false;

    for (int iteration = 1; ((maxIterations < 1) || (iteration <= maxIterations)) && !converged; iteration++) {
      log.info("LDA Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path stateOut = new Path(output, "state-" + iteration);
      double ll = runIteration(conf, input, stateIn, stateOut, numTopics, numWords, topicSmoothing);
      double relChange = (oldLL - ll) / oldLL;

      // now point the input to the old output directory
      log.info("Iteration {} finished. Log Likelihood: {}", iteration, ll);
      log.info("(Old LL: {})", oldLL);
      log.info("(Rel Change: {})", relChange);

      converged = (iteration > 3) && (relChange < OVERALL_CONVERGENCE);
      stateIn = stateOut;
      oldLL = ll;
    }
  }

  private static void writeInitialState(Path statePath, int numTopics, int numWords) throws IOException {
    Configuration job = new Configuration();
    FileSystem fs = statePath.getFileSystem(job);

    DoubleWritable v = new DoubleWritable();

    Random random = RandomUtils.getRandom();

    for (int k = 0; k < numTopics; ++k) {
      Path path = new Path(statePath, "part-" + k);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, IntPairWritable.class, DoubleWritable.class);

      double total = 0.0; // total number of pseudo counts we made
      for (int w = 0; w < numWords; ++w) {
        Writable kw = new IntPairWritable(k, w);
        // A small amount of random noise, minimized by having a floor.
        double pseudocount = random.nextDouble() + 1.0E-8;
        total += pseudocount;
        v.set(Math.log(pseudocount));
        writer.append(kw, v);
      }
      Writable kTsk = new IntPairWritable(k, TOPIC_SUM_KEY);
      v.set(Math.log(total));
      writer.append(kTsk, v);

      writer.close();
    }
  }

  private static double findLL(Path statePath, Configuration job) throws IOException {
    FileSystem fs = statePath.getFileSystem(job);

    double ll = 0.0;

    IntPairWritable key = new IntPairWritable();
    DoubleWritable value = new DoubleWritable();
    for (FileStatus status : fs.globStatus(new Path(statePath, "part-*"))) {
      Path path = status.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
      while (reader.next(key, value)) {
        if (key.getFirst() == LOG_LIKELIHOOD_KEY) {
          ll = value.get();
          break;
        }
      }
      reader.close();
    }

    return ll;
  }

  /**
   * Run the job using supplied arguments
   * @param conf TODO
   * @param input
   *          the directory pathname for input points
   * @param stateIn
   *          the directory pathname for input state
   * @param stateOut
   *          the directory pathname for output state
   * @param numTopics
   *          the number of clusters
   */
  private static double runIteration(Configuration conf,
                                     Path input,
                                     Path stateIn,
                                     Path stateOut,
                                     int numTopics,
                                     int numWords,
                                     double topicSmoothing)
    throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(STATE_IN_KEY, stateIn.toString());
    conf.set(NUM_TOPICS_KEY, Integer.toString(numTopics));
    conf.set(NUM_WORDS_KEY, Integer.toString(numWords));
    conf.set(TOPIC_SMOOTHING_KEY, Double.toString(topicSmoothing));

    Job job = new Job(conf);

    job.setOutputKeyClass(IntPairWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
    FileInputFormat.addInputPaths(job, input.toString());
    FileOutputFormat.setOutputPath(job, stateOut);

    job.setMapperClass(LDAMapper.class);
    job.setReducerClass(LDAReducer.class);
    job.setCombinerClass(LDAReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setJarByClass(LDADriver.class);

    job.waitForCompletion(true);
    return findLL(stateOut, conf);
  }
}
