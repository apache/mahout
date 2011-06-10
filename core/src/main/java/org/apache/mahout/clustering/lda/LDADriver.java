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

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

/**
 * Estimates an LDA model from a corpus of documents, which are SparseVectors of word counts. At each phase,
 * it outputs a matrix of log probabilities of each topic.
 */
public final class LDADriver extends AbstractJob {

  private static final String TOPIC_SMOOTHING_OPTION = "topicSmoothing";
  private static final String NUM_WORDS_OPTION = "numWords";
  private static final String NUM_TOPICS_OPTION = "numTopics";
  // TODO: sequential iteration is not yet correct.
  // private static final String SEQUENTIAL_OPTION = "sequential";
  static final String STATE_IN_KEY = "org.apache.mahout.clustering.lda.stateIn";
  static final String NUM_TOPICS_KEY = "org.apache.mahout.clustering.lda.numTopics";
  static final String NUM_WORDS_KEY = "org.apache.mahout.clustering.lda.numWords";
  static final String TOPIC_SMOOTHING_KEY = "org.apache.mahout.clustering.lda.topicSmoothing";
  static final int LOG_LIKELIHOOD_KEY = -2;
  static final int TOPIC_SUM_KEY = -1;
  static final double OVERALL_CONVERGENCE = 1.0E-5;

  private static final Logger log = LoggerFactory.getLogger(LDADriver.class);

  private LDAState state = null;

  private LDAInference inference = null;

  private Iterable<Pair<Writable, VectorWritable>> trainingCorpus = null;

  private LDADriver() {
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new LDADriver(), args);
  }

  public static LDAState createState(Configuration job) {
    return createState(job, false);
  }

  public static LDAState createState(Configuration job, boolean empty) {
    String statePath = job.get(STATE_IN_KEY);
    int numTopics = Integer.parseInt(job.get(NUM_TOPICS_KEY));
    int numWords = Integer.parseInt(job.get(NUM_WORDS_KEY));
    double topicSmoothing = Double.parseDouble(job.get(TOPIC_SMOOTHING_KEY));

    Path dir = new Path(statePath);

    // TODO scalability bottleneck: numWords * numTopics * 8bytes for the driver *and* M/R classes
    DenseMatrix pWgT = new DenseMatrix(numTopics, numWords);
    double[] logTotals = new double[numTopics];
    Arrays.fill(logTotals, Double.NEGATIVE_INFINITY);
    double ll = 0.0;
    if (empty) {
      return new LDAState(numTopics, numWords, topicSmoothing, pWgT, logTotals, ll);
    }
    for (Pair<IntPairWritable,DoubleWritable> record
         : new SequenceFileDirIterable<IntPairWritable, DoubleWritable>(new Path(dir, "part-*"),
                                                                        PathType.GLOB,
                                                                        null,
                                                                        null,
                                                                        true,
                                                                        job)) {
      IntPairWritable key = record.getFirst();
      DoubleWritable value = record.getSecond();
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
    // addOption(SEQUENTIAL_OPTION, "seq", "Run sequentially (not Hadoop-based).  Default is false.", "false");
    addOption(DefaultOptionCreator.maxIterationsOption().withRequired(false).create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path output = getOutputPath();
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    int numTopics = Integer.parseInt(getOption(NUM_TOPICS_OPTION));
    int numWords = Integer.parseInt(getOption(NUM_WORDS_OPTION));
    double topicSmoothing = Double.parseDouble(getOption(TOPIC_SMOOTHING_OPTION));
    if (topicSmoothing < 1) {
      topicSmoothing = 50.0 / numTopics;
    }
    boolean runSequential = false; // Boolean.parseBoolean(getOption(SEQUENTIAL_OPTION));

    run(getConf(), input, output, numTopics, numWords, topicSmoothing, maxIterations, runSequential);

    return 0;
  }

  private static Path getLastKnownStatePath(Configuration conf, Path stateDir) throws IOException {
    FileSystem fs = FileSystem.get(conf);
    Path lastPath = null;
    int maxIteration = Integer.MIN_VALUE;
    for (FileStatus fstatus : fs.globStatus(new Path(stateDir, "state-*"))) {
      try {
        int iteration = Integer.parseInt(fstatus.getPath().getName().split("-")[1]);
        if(iteration > maxIteration) {
          maxIteration = iteration;
          lastPath = fstatus.getPath();
        }
      } catch(NumberFormatException nfe) {
        throw new IOException(nfe);
      }
    }
    return lastPath;
  }

  private void run(Configuration conf,
                          Path input,
                          Path output,
                          int numTopics,
                          int numWords,
                          double topicSmoothing,
                          int maxIterations,
                          boolean runSequential)
    throws IOException, InterruptedException, ClassNotFoundException {
    Path lastKnownState = getLastKnownStatePath(conf, output);
    Path stateIn;
    if (lastKnownState == null) {
      stateIn = new Path(output, "state-0");
      writeInitialState(stateIn, numTopics, numWords);
    } else {
      stateIn = lastKnownState;
    }
    conf.set(STATE_IN_KEY, stateIn.toString());
    conf.set(NUM_TOPICS_KEY, Integer.toString(numTopics));
    conf.set(NUM_WORDS_KEY, Integer.toString(numWords));
    conf.set(TOPIC_SMOOTHING_KEY, Double.toString(topicSmoothing));
    double oldLL = Double.NEGATIVE_INFINITY;
    boolean converged = false;
    int iteration = Integer.parseInt(stateIn.getName().split("-")[1]) + 1;
    for (; ((maxIterations < 1) || (iteration <= maxIterations)) && !converged; iteration++) {
      log.info("LDA Iteration {}", iteration);
      conf.set(STATE_IN_KEY, stateIn.toString());
      // point the output to a new directory per iteration
      Path stateOut = new Path(output, "state-" + iteration);
      double ll = runSequential
          ? runIterationSequential(conf, input, stateOut)
          : runIteration(conf, input, stateIn, stateOut);
      double relChange = (oldLL - ll) / oldLL;

      // now point the input to the old output directory
      log.info("Iteration {} finished. Log Likelihood: {}", iteration, ll);
      log.info("(Old LL: {})", oldLL);
      log.info("(Rel Change: {})", relChange);

      converged = (iteration > 3) && (relChange < OVERALL_CONVERGENCE);
      stateIn = stateOut;
      oldLL = ll;
    }
    if(runSequential) {
      computeDocumentTopicProbabilitiesSequential(conf, input, new Path(output, "docTopics"));
    } else {
      computeDocumentTopicProbabilities(conf,
                                        input,
                                        stateIn,
                                        new Path(output, "docTopics"),
                                        numTopics,
                                        numWords,
                                        topicSmoothing);
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

      try {
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
      } finally {
        Closeables.closeQuietly(writer);
      }
    }
  }

  private static void writeState(Configuration job, LDAState state, Path statePath) throws IOException {
    FileSystem fs = statePath.getFileSystem(job);
    DoubleWritable v = new DoubleWritable();

    for (int k = 0; k < state.getNumTopics(); ++k) {
      Path path = new Path(statePath, "part-" + k);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, IntPairWritable.class, DoubleWritable.class);

      try {
        for (int w = 0; w < state.getNumWords(); ++w) {
          Writable kw = new IntPairWritable(k, w);
          v.set(state.logProbWordGivenTopic(w,k) + state.getLogTotal(k));
          writer.append(kw, v);
        }
        Writable kTsk = new IntPairWritable(k, TOPIC_SUM_KEY);
        v.set(state.getLogTotal(k));
        writer.append(kTsk, v);
      } finally {
        Closeables.closeQuietly(writer);
      }
    }
    Path path = new Path(statePath, "part-" + LOG_LIKELIHOOD_KEY);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path, IntPairWritable.class, DoubleWritable.class);
    try {
      Writable kTsk = new IntPairWritable(LOG_LIKELIHOOD_KEY,LOG_LIKELIHOOD_KEY);
      v.set(state.getLogLikelihood());
      writer.append(kTsk, v);
    } finally {
      Closeables.closeQuietly(writer);
    }
  }

  private static double findLL(Path statePath, Configuration job) throws IOException {
    FileSystem fs = statePath.getFileSystem(job);
    double ll = 0.0;
    for (FileStatus status : fs.globStatus(new Path(statePath, "part-*"))) {
      Path path = status.getPath();
      SequenceFileIterator<IntPairWritable,DoubleWritable> iterator =
          new SequenceFileIterator<IntPairWritable,DoubleWritable>(path, true, job);
      try {
        while (iterator.hasNext()) {
          Pair<IntPairWritable,DoubleWritable> record = iterator.next();
          if (record.getFirst().getFirst() == LOG_LIKELIHOOD_KEY) {
            ll = record.getSecond().get();
            break;
          }
        }
      } finally {
        Closeables.closeQuietly(iterator);
      }
    }
    return ll;
  }

  private double runIterationSequential(Configuration conf, Path input, Path stateOut) throws IOException {
    if (state == null) {
      state = createState(conf);
    }
    if (trainingCorpus == null) {
      Class<? extends Writable> keyClass = peekAtSequenceFileForKeyType(conf, input);
      Collection<Pair<Writable, VectorWritable>> corpus = new LinkedList<Pair<Writable, VectorWritable>>();
      for (FileStatus fileStatus : FileSystem.get(conf).globStatus(new Path(input, "part-*"))) {
        Path inputPart = fileStatus.getPath();
        SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf), inputPart, conf);
        Writable key = ReflectionUtils.newInstance(keyClass, conf);
        VectorWritable value = new VectorWritable();
        while (reader.next(key, value)) {
          Writable nextKey = ReflectionUtils.newInstance(keyClass, conf);
          VectorWritable nextValue = new VectorWritable();
          corpus.add(new Pair<Writable,VectorWritable>(key, value));
          key = nextKey;
          value = nextValue;
        }
      }
      trainingCorpus = corpus;
    }
    if (inference == null) {
      inference = new LDAInference(state);
    }
    LDAState newState = createState(conf, true);
    double ll = 0.0;
    for (Pair<Writable, VectorWritable> slice : trainingCorpus) {
      LDAInference.InferredDocument doc;
      Vector wordCounts = slice.getSecond().get();
      try {
        doc = inference.infer(wordCounts);
      } catch (ArrayIndexOutOfBoundsException e1) {
        throw new IllegalStateException(
         "This is probably because the --numWords argument is set too small.  \n"
         + "\tIt needs to be >= than the number of words (terms actually) in the corpus and can be \n"
         + "\tlarger if some storage inefficiency can be tolerated.", e1);
      }

      for (Iterator<Vector.Element> iter = wordCounts.iterateNonZero(); iter.hasNext();) {
        Vector.Element e = iter.next();
        int w = e.index();

        for (int k = 0; k < state.getNumTopics(); ++k) {
          double vwUpdate = doc.phi(k, w) + Math.log(e.get());
          newState.updateLogProbGivenTopic(w, k, vwUpdate); // update state.topicWordProbabilities[v,w]!
          newState.updateLogTotals(k, vwUpdate);
        }
        ll += doc.getLogLikelihood();
      }
    }
    newState.setLogLikelihood(ll);
    writeState(conf, newState, stateOut);
    state = newState;
    newState = null;

    return ll;
  }

  /**
   * Run the job using supplied arguments
   * @param input
   *          the directory pathname for input points
   * @param stateIn
   *          the directory pathname for input state
   * @param stateOut
   *          the directory pathname for output state
   */
  private static double runIteration(Configuration conf,
                                     Path input,
                                     Path stateIn,
                                     Path stateOut)
    throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(STATE_IN_KEY, stateIn.toString());

    Job job = new Job(conf, "LDA Driver running runIteration over stateIn: " + stateIn);
    job.setOutputKeyClass(IntPairWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
    FileInputFormat.addInputPaths(job, input.toString());
    FileOutputFormat.setOutputPath(job, stateOut);

    job.setMapperClass(LDAWordTopicMapper.class);
    job.setReducerClass(LDAReducer.class);
    job.setCombinerClass(LDAReducer.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setJarByClass(LDADriver.class);

    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("LDA Iteration failed processing " + stateIn);
    }
    return findLL(stateOut, conf);
  }

  private static void computeDocumentTopicProbabilities(Configuration conf,
                                                        Path input,
                                                        Path stateIn,
                                                        Path outputPath,
                                                        int numTopics,
                                                        int numWords,
                                                        double topicSmoothing)
    throws IOException, InterruptedException, ClassNotFoundException {
    conf.set(STATE_IN_KEY, stateIn.toString());
    conf.set(NUM_TOPICS_KEY, Integer.toString(numTopics));
    conf.set(NUM_WORDS_KEY, Integer.toString(numWords));
    conf.set(TOPIC_SMOOTHING_KEY, Double.toString(topicSmoothing));

    Job job = new Job(conf, "LDA Driver computing p(topic|doc) for all docs/topics with stateIn: " + stateIn);
    job.setOutputKeyClass(peekAtSequenceFileForKeyType(conf, input));
    job.setOutputValueClass(VectorWritable.class);
    FileInputFormat.addInputPaths(job, input.toString());
    FileOutputFormat.setOutputPath(job, outputPath);

    job.setMapperClass(LDADocumentTopicMapper.class);
    job.setNumReduceTasks(0);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setJarByClass(LDADriver.class);

    if (!job.waitForCompletion(true)) {
      throw new InterruptedException("LDA failed to compute and output document topic probabilities with: "+ stateIn);
    }
  }

  private void computeDocumentTopicProbabilitiesSequential(Configuration conf, Path input, Path outputPath)
    throws IOException {
    FileSystem fs = input.getFileSystem(conf);
    Class<? extends Writable> keyClass = peekAtSequenceFileForKeyType(conf, input);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, outputPath, keyClass, VectorWritable.class);

    try {
      Writable key = ReflectionUtils.newInstance(keyClass, conf);
      Writable vw = new VectorWritable();

      for (Pair<Writable, VectorWritable> slice : trainingCorpus) {
        Vector wordCounts = slice.getSecond().get();
        try {
          inference.infer(wordCounts);
        } catch (ArrayIndexOutOfBoundsException e1) {
          throw new IllegalStateException(
           "This is probably because the --numWords argument is set too small.  \n"
           + "\tIt needs to be >= than the number of words (terms actually) in the corpus and can be \n"
           + "\tlarger if some storage inefficiency can be tolerated.", e1);
        }
        writer.append(key, vw);
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }

  private static Class<? extends Writable> peekAtSequenceFileForKeyType(Configuration conf, Path input) {
    try {
      SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.get(conf), input, conf);
      return (Class<? extends Writable>) reader.getKeyClass();
    } catch (IOException ioe) {
      return Text.class;
    }
  }
}
