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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.matrix.DenseMatrix;
import org.apache.mahout.utils.CommandLineUtil;
import org.apache.mahout.utils.HadoopUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
* Estimates an LDA model from a corpus of documents,
* which are SparseVectors of word counts. At each
* phase, it outputs a matrix of log probabilities of
* each topic. 
*/
public final class LDADriver {

  static final String STATE_IN_KEY = "org.apache.mahout.clustering.lda.stateIn";

  static final String NUM_TOPICS_KEY = "org.apache.mahout.clustering.lda.numTopics";
  static final String NUM_WORDS_KEY = "org.apache.mahout.clustering.lda.numWords";

  static final String TOPIC_SMOOTHING_KEY = "org.apache.mahout.clustering.lda.topicSmoothing";

  static final int LOG_LIKELIHOOD_KEY = -2;
  static final int TOPIC_SUM_KEY = -1;

  static final double OVERALL_CONVERGENCE = 1E-5;

  private static final Logger log = LoggerFactory.getLogger(LDADriver.class);

  private LDADriver() {
  }

  public static void main(String[] args) throws InstantiationException,
      IllegalAccessException, ClassNotFoundException,
      IOException, InterruptedException {

    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withRequired(true).withArgument(
        abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Path for input Vectors. Must be a SequenceFile of Writable, Vector").withShortName("i").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
        "The Output Working Directory").withShortName("o").create();

    Option overwriteOutput = obuilder.withLongName("overwrite").withRequired(false).withDescription(
        "If set, overwrite the output directory").withShortName("w").create();

    Option topicsOpt = obuilder.withLongName("numTopics").withRequired(true).withArgument(
        abuilder.withName("numTopics").withMinimum(1).withMaximum(1).create()).withDescription(
        "The number of topics").withShortName("k").create();

    Option wordsOpt = obuilder.withLongName("numWords").withRequired(true).withArgument(
        abuilder.withName("numWords").withMinimum(1).withMaximum(1).create()).withDescription(
        "The number of words in the corpus").withShortName("v").create();

    Option topicSmOpt = obuilder.withLongName("topicSmoothing").withRequired(false).withArgument(abuilder
        .withName("topicSmoothing").withDefault(-1.0).withMinimum(0).withMaximum(1).create()).withDescription(
        "Topic smoothing parameter. Default is 50/numTopics.").withShortName("a").create();

    Option maxIterOpt = obuilder.withLongName("maxIter").withRequired(false).withArgument(
        abuilder.withName("maxIter").withDefault(-1).withMinimum(0).withMaximum(1).create()).withDescription(
        "Max iterations to run (or until convergence). -1 (default) waits until convergence.").create();

    Option numReducOpt = obuilder.withLongName("numReducers").withRequired(false).withArgument(
        abuilder.withName("numReducers").withDefault(10).withMinimum(0).withMaximum(1).create()).withDescription(
        "Max iterations to run (or until convergence). Default 10").create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(
        topicsOpt).withOption(wordsOpt).withOption(topicSmOpt).withOption(maxIterOpt).withOption(
            numReducOpt).withOption(overwriteOutput).withOption(helpOpt).create();
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      String input = cmdLine.getValue(inputOpt).toString();
      String output = cmdLine.getValue(outputOpt).toString();

      int maxIterations = -1;
      if (cmdLine.hasOption(maxIterOpt)) {
        maxIterations = Integer.parseInt(cmdLine.getValue(maxIterOpt).toString());
      }

      int numReduceTasks = 2;
      if (cmdLine.hasOption(numReducOpt)) {
        numReduceTasks = Integer.parseInt(cmdLine.getValue(numReducOpt).toString());
      }

      int numTopics = 20;
      if (cmdLine.hasOption(topicsOpt)) {
        numTopics = Integer.parseInt(cmdLine.getValue(topicsOpt).toString());
      }

      int numWords = 20;
      if (cmdLine.hasOption(wordsOpt)) {
        numWords = Integer.parseInt(cmdLine.getValue(wordsOpt).toString());
      }

      if (cmdLine.hasOption(overwriteOutput)) {
        HadoopUtil.overwriteOutput(output);
      }

      double topicSmoothing = -1.0;
      if (cmdLine.hasOption(topicSmOpt)) {
        topicSmoothing = Double.parseDouble(cmdLine.getValue(maxIterOpt).toString());
      }
      if(topicSmoothing < 1) {
        topicSmoothing = 50. / numTopics;
      }

      runJob(input, output, numTopics, numWords, topicSmoothing, maxIterations,
          numReduceTasks);

    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }

  /**
   * Run the job using supplied arguments
   *
   * @param input           the directory pathname for input points
   * @param output          the directory pathname for output points
   * @param numTopics       the number of topics
   * @param numWords        the number of words
   * @param topicSmoothing  pseudocounts for each topic, typically small &lt; .5
   * @param maxIterations   the maximum number of iterations
   * @param numReducers     the number of Reducers desired
   * @throws IOException 
   */
  public static void runJob(String input, String output, int numTopics, 
      int numWords, double topicSmoothing, int maxIterations, int numReducers)
      throws IOException, InterruptedException, ClassNotFoundException {

    String stateIn = output + "/state-0";
    writeInitialState(stateIn, numTopics, numWords);
    double oldLL = Double.NEGATIVE_INFINITY;
    boolean converged = false;

    for (int iteration = 0; (maxIterations < 1 || iteration < maxIterations) && !converged; iteration++) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      String stateOut = output + "/state-" + (iteration + 1);
      double ll = runIteration(input, stateIn, stateOut, numTopics,
          numWords, topicSmoothing, numReducers);
      double relChange = (oldLL - ll) / oldLL;

      // now point the input to the old output directory
      log.info("Iteration {} finished. Log Likelihood: {}", iteration, ll);
      log.info("(Old LL: {})", oldLL);
      log.info("(Rel Change: {})", relChange);

      converged = iteration > 2 && relChange < OVERALL_CONVERGENCE;
      stateIn = stateOut;
      oldLL = ll;
    }
  }

  private static void writeInitialState(String statePath,  
      int numTopics, int numWords) throws IOException {
    Path dir = new Path(statePath);
    Configuration job = new Configuration();
    FileSystem fs = dir.getFileSystem(job);

    IntPairWritable kw = new IntPairWritable();
    DoubleWritable v = new DoubleWritable();

    Random random = new Random();

    for (int k = 0; k < numTopics; ++k) {
      Path path = new Path(dir, "part-" + k);
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, job, path,
          IntPairWritable.class, DoubleWritable.class);

      double total = 0.0; // total number of pseudo counts we made

      kw.setX(k);
      for (int w = 0; w < numWords; ++w) {
        kw.setY(w);
        // A small amount of random noise, minimized by having a floor.
        double pseudocount = random.nextDouble() + 1E-8;
        total += pseudocount;
        v.set(Math.log(pseudocount));
        writer.append(kw, v);
      }

      kw.setY(TOPIC_SUM_KEY);
      v.set(Math.log(total));
      writer.append(kw, v);

      writer.close();
    }
  }

  private static double findLL(String statePath, Configuration job) throws IOException {
    Path dir = new Path(statePath);
    FileSystem fs = dir.getFileSystem(job);

    double ll = 0.0;

    IntPairWritable key = new IntPairWritable();
    DoubleWritable value = new DoubleWritable();
    for (FileStatus status : fs.globStatus(new Path(dir, "*"))) { 
      Path path = status.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
      while (reader.next(key, value)) {
        if (key.getX() == LOG_LIKELIHOOD_KEY) {
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
   *
   * @param input         the directory pathname for input points
   * @param stateIn       the directory pathname for input state
   * @param stateOut      the directory pathname for output state
   * @param modelFactory  the class name of the model factory class
   * @param numTopics   the number of clusters
   * @param alpha_0       alpha_0
   * @param numReducers   the number of Reducers desired
   */
  public static double runIteration(String input, String stateIn,
      String stateOut, int numTopics, int numWords, double topicSmoothing,
      int numReducers) throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(STATE_IN_KEY, stateIn);
    conf.set(NUM_TOPICS_KEY, Integer.toString(numTopics));
    conf.set(NUM_WORDS_KEY, Integer.toString(numWords));
    conf.set(TOPIC_SMOOTHING_KEY, Double.toString(topicSmoothing));

    Job job = new Job(conf);

    job.setOutputKeyClass(IntPairWritable.class);
    job.setOutputValueClass(DoubleWritable.class);

    FileInputFormat.addInputPaths(job, input);
    Path outPath = new Path(stateOut);
    FileOutputFormat.setOutputPath(job, outPath);

    job.setMapperClass(LDAMapper.class);
    job.setReducerClass(LDAReducer.class);
    job.setCombinerClass(LDAReducer.class);
    job.setNumReduceTasks(numReducers);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);

    job.waitForCompletion(true);
    return findLL(stateOut, conf);
  }

  static LDAState createState(Configuration job) throws IOException {
    String statePath = job.get(LDADriver.STATE_IN_KEY);
    int numTopics = Integer.parseInt(job.get(LDADriver.NUM_TOPICS_KEY));
    int numWords = Integer.parseInt(job.get(LDADriver.NUM_WORDS_KEY));
    double topicSmoothing = Double.parseDouble(job.get(LDADriver.TOPIC_SMOOTHING_KEY));

    Path dir = new Path(statePath);
    FileSystem fs = dir.getFileSystem(job);

    DenseMatrix pWgT = new DenseMatrix(numTopics, numWords);
    double[] logTotals = new double[numTopics];
    double ll = 0.0;

    IntPairWritable key = new IntPairWritable();
    DoubleWritable value = new DoubleWritable();
    for (FileStatus status : fs.globStatus(new Path(dir, "*"))) { 
      Path path = status.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, job);
      while (reader.next(key, value)) {
        int topic = key.getX();
        int word = key.getY();
        if (word == TOPIC_SUM_KEY) {
          logTotals[topic] = value.get();
          assert !Double.isInfinite(value.get());
        } else if (topic == LOG_LIKELIHOOD_KEY) {
          ll = value.get();
        } else {
          //System.out.println(topic + " " + word);
          assert topic >= 0 && word >= 0 : topic + " " + word;
          assert pWgT.getQuick(topic, word) == 0.0;
          pWgT.setQuick(topic, word, value.get());
          assert !Double.isInfinite(pWgT.getQuick(topic, word));
        }
      }
      reader.close();
    }

    return new LDAState(numTopics, numWords, topicSmoothing,
        pWgT, logTotals, ll);
  }
}
