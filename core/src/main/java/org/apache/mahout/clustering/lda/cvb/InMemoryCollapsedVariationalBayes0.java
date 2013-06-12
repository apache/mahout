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
import com.google.common.collect.Maps;
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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.DistributedRowMatrixWriter;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.NamedVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Runs the same algorithm as {@link CVB0Driver}, but sequentially, in memory.  Memory requirements
 * are currently: the entire corpus is read into RAM, two copies of the model (each of size
 * numTerms * numTopics), and another matrix of size numDocs * numTopics is held in memory
 * (to store p(topic|doc) for all docs).
 *
 * But if all this fits in memory, this can be significantly faster than an iterative MR job.
 */
public class InMemoryCollapsedVariationalBayes0 extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(InMemoryCollapsedVariationalBayes0.class);

  private int numTopics;
  private int numTerms;
  private int numDocuments;
  private double alpha;
  private double eta;
  //private int minDfCt;
  //private double maxDfPct;
  private boolean verbose = false;
  private String[] terms;  // of length numTerms;
  private Matrix corpusWeights; // length numDocs;
  private double totalCorpusWeight;
  private double initialModelCorpusFraction;
  private Matrix docTopicCounts;
  private int numTrainingThreads;
  private int numUpdatingThreads;
  private ModelTrainer modelTrainer;

  private InMemoryCollapsedVariationalBayes0() {
    // only for main usage
  }

  public void setVerbose(boolean verbose) {
    this.verbose = verbose;
  }
    
  public InMemoryCollapsedVariationalBayes0(Matrix corpus,
                                            String[] terms,
                                            int numTopics,
                                            double alpha,
                                            double eta,
                                            int numTrainingThreads,
                                            int numUpdatingThreads,
                                            double modelCorpusFraction) {
    //this.seed = seed;
    this.numTopics = numTopics;
    this.alpha = alpha;
    this.eta = eta;
    //this.minDfCt = 0;
    //this.maxDfPct = 1.0f;
    corpusWeights = corpus;
    numDocuments = corpus.numRows();
    this.terms = terms;
    this.initialModelCorpusFraction = modelCorpusFraction;
    numTerms = terms != null ? terms.length : corpus.numCols();
    Map<String, Integer> termIdMap = Maps.newHashMap();
    if (terms != null) {
      for (int t = 0; t < terms.length; t++) {
        termIdMap.put(terms[t], t);
      }
    }
    this.numTrainingThreads = numTrainingThreads;
    this.numUpdatingThreads = numUpdatingThreads;
    postInitCorpus();
    initializeModel();
  }

  private void postInitCorpus() {
    totalCorpusWeight = 0;
    int numNonZero = 0;
    for (int i = 0; i < numDocuments; i++) {
      Vector v = corpusWeights.viewRow(i);
      double norm;
      if (v != null && (norm = v.norm(1)) != 0) {
        numNonZero += v.getNumNondefaultElements();
        totalCorpusWeight += norm;
      }
    }
    String s = "Initializing corpus with %d docs, %d terms, %d nonzero entries, total termWeight %f";
    log.info(String.format(s, numDocuments, numTerms, numNonZero, totalCorpusWeight));
  }

  private void initializeModel() {
    TopicModel topicModel = new TopicModel(numTopics, numTerms, eta, alpha, RandomUtils.getRandom(), terms,
        numUpdatingThreads, initialModelCorpusFraction == 0 ? 1 : initialModelCorpusFraction * totalCorpusWeight);
    topicModel.setConf(getConf());

    TopicModel updatedModel = initialModelCorpusFraction == 0
        ? new TopicModel(numTopics, numTerms, eta, alpha, null, terms, numUpdatingThreads, 1)
        : topicModel;
    updatedModel.setConf(getConf());
    docTopicCounts = new DenseMatrix(numDocuments, numTopics);
    docTopicCounts.assign(1.0 / numTopics);
    modelTrainer = new ModelTrainer(topicModel, updatedModel, numTrainingThreads, numTopics, numTerms);
  }

  /*
  private void inferDocuments(double convergence, int maxIter, boolean recalculate) {
    for (int docId = 0; docId < corpusWeights.numRows() ; docId++) {
      Vector inferredDocument = topicModel.infer(corpusWeights.viewRow(docId),
          docTopicCounts.viewRow(docId));
      // do what now?
    }
  }
   */

  public void trainDocuments() {
    trainDocuments(0);
  }

  public void trainDocuments(double testFraction) {
    long start = System.nanoTime();
    modelTrainer.start();
    for (int docId = 0; docId < corpusWeights.numRows(); docId++) {
      if (testFraction == 0 || docId % (1 / testFraction) != 0) {
        Vector docTopics = new DenseVector(numTopics).assign(1.0 / numTopics); // docTopicCounts.getRow(docId)
        modelTrainer.trainSync(corpusWeights.viewRow(docId), docTopics , true, 10);
      }
    }
    modelTrainer.stop();
    logTime("train documents", System.nanoTime() - start);
  }

  /*
  private double error(int docId) {
    Vector docTermCounts = corpusWeights.viewRow(docId);
    if (docTermCounts == null) {
      return 0;
    } else {
      Vector expectedDocTermCounts =
          topicModel.infer(corpusWeights.viewRow(docId), docTopicCounts.viewRow(docId));
      double expectedNorm = expectedDocTermCounts.norm(1);
      return expectedDocTermCounts.times(docTermCounts.norm(1)/expectedNorm)
          .minus(docTermCounts).norm(1);
    }
  }

  private double error() {
    long time = System.nanoTime();
    double error = 0;
    for (int docId = 0; docId < numDocuments; docId++) {
      error += error(docId);
    }
    logTime("error calculation", System.nanoTime() - time);
    return error / totalCorpusWeight;
  }
   */

  public double iterateUntilConvergence(double minFractionalErrorChange,
      int maxIterations, int minIter) {
    return iterateUntilConvergence(minFractionalErrorChange, maxIterations, minIter, 0);
  }

  public double iterateUntilConvergence(double minFractionalErrorChange,
      int maxIterations, int minIter, double testFraction) {
    int iter = 0;
    double oldPerplexity = 0;
    while (iter < minIter) {
      trainDocuments(testFraction);
      if (verbose) {
        log.info("model after: {}: {}", iter, modelTrainer.getReadModel());
      }
      log.info("iteration {} complete", iter);
      oldPerplexity = modelTrainer.calculatePerplexity(corpusWeights, docTopicCounts,
          testFraction);
      log.info("{} = perplexity", oldPerplexity);
      iter++;
    }
    double newPerplexity = 0;
    double fractionalChange = Double.MAX_VALUE;
    while (iter < maxIterations && fractionalChange > minFractionalErrorChange) {
      trainDocuments();
      if (verbose) {
        log.info("model after: {}: {}", iter, modelTrainer.getReadModel());
      }
      newPerplexity = modelTrainer.calculatePerplexity(corpusWeights, docTopicCounts,
          testFraction);
      log.info("{} = perplexity", newPerplexity);
      iter++;
      fractionalChange = Math.abs(newPerplexity - oldPerplexity) / oldPerplexity;
      log.info("{} = fractionalChange", fractionalChange);
      oldPerplexity = newPerplexity;
    }
    if (iter < maxIterations) {
      log.info(String.format("Converged! fractional error change: %f, error %f",
          fractionalChange, newPerplexity));
    } else {
      log.info(String.format("Reached max iteration count (%d), fractional error change: %f, error: %f",
          maxIterations, fractionalChange, newPerplexity));
    }
    return newPerplexity;
  }

  public void writeModel(Path outputPath) throws IOException {
    modelTrainer.persist(outputPath);
  }

  private static void logTime(String label, long nanos) {
    log.info("{} time: {}ms", label, nanos / 1.0e6);
  }

  public static int main2(String[] args, Configuration conf) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option helpOpt = DefaultOptionCreator.helpOption();

    Option inputDirOpt = obuilder.withLongName("input").withRequired(true).withArgument(
      abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
      "The Directory on HDFS containing the collapsed, properly formatted files having "
          + "one doc per line").withShortName("i").create();

    Option dictOpt = obuilder.withLongName("dictionary").withRequired(false).withArgument(
      abuilder.withName("dictionary").withMinimum(1).withMaximum(1).create()).withDescription(
      "The path to the term-dictionary format is ... ").withShortName("d").create();

    Option dfsOpt = obuilder.withLongName("dfs").withRequired(false).withArgument(
      abuilder.withName("dfs").withMinimum(1).withMaximum(1).create()).withDescription(
      "HDFS namenode URI").withShortName("dfs").create();

    Option numTopicsOpt = obuilder.withLongName("numTopics").withRequired(true).withArgument(abuilder
        .withName("numTopics").withMinimum(1).withMaximum(1)
        .create()).withDescription("Number of topics to learn").withShortName("top").create();

    Option outputTopicFileOpt = obuilder.withLongName("topicOutputFile").withRequired(true).withArgument(
        abuilder.withName("topicOutputFile").withMinimum(1).withMaximum(1).create())
        .withDescription("File to write out p(term | topic)").withShortName("to").create();

    Option outputDocFileOpt = obuilder.withLongName("docOutputFile").withRequired(true).withArgument(
        abuilder.withName("docOutputFile").withMinimum(1).withMaximum(1).create())
        .withDescription("File to write out p(topic | docid)").withShortName("do").create();

    Option alphaOpt = obuilder.withLongName("alpha").withRequired(false).withArgument(abuilder
        .withName("alpha").withMinimum(1).withMaximum(1).withDefault("0.1").create())
        .withDescription("Smoothing parameter for p(topic | document) prior").withShortName("a").create();

    Option etaOpt = obuilder.withLongName("eta").withRequired(false).withArgument(abuilder
        .withName("eta").withMinimum(1).withMaximum(1).withDefault("0.1").create())
        .withDescription("Smoothing parameter for p(term | topic)").withShortName("e").create();

    Option maxIterOpt = obuilder.withLongName("maxIterations").withRequired(false).withArgument(abuilder
        .withName("maxIterations").withMinimum(1).withMaximum(1).withDefault("10").create())
        .withDescription("Maximum number of training passes").withShortName("m").create();

    Option modelCorpusFractionOption = obuilder.withLongName("modelCorpusFraction")
        .withRequired(false).withArgument(abuilder.withName("modelCorpusFraction").withMinimum(1)
        .withMaximum(1).withDefault("0.0").create()).withShortName("mcf")
        .withDescription("For online updates, initial value of |model|/|corpus|").create();

    Option burnInOpt = obuilder.withLongName("burnInIterations").withRequired(false).withArgument(abuilder
        .withName("burnInIterations").withMinimum(1).withMaximum(1).withDefault("5").create())
        .withDescription("Minimum number of iterations").withShortName("b").create();

    Option convergenceOpt = obuilder.withLongName("convergence").withRequired(false).withArgument(abuilder
        .withName("convergence").withMinimum(1).withMaximum(1).withDefault("0.0").create())
        .withDescription("Fractional rate of perplexity to consider convergence").withShortName("c").create();

    Option reInferDocTopicsOpt = obuilder.withLongName("reInferDocTopics").withRequired(false)
        .withArgument(abuilder.withName("reInferDocTopics").withMinimum(1).withMaximum(1)
        .withDefault("no").create())
        .withDescription("re-infer p(topic | doc) : [no | randstart | continue]")
        .withShortName("rdt").create();

    Option numTrainThreadsOpt = obuilder.withLongName("numTrainThreads").withRequired(false)
        .withArgument(abuilder.withName("numTrainThreads").withMinimum(1).withMaximum(1)
        .withDefault("1").create())
        .withDescription("number of threads to train with")
        .withShortName("ntt").create();

    Option numUpdateThreadsOpt = obuilder.withLongName("numUpdateThreads").withRequired(false)
        .withArgument(abuilder.withName("numUpdateThreads").withMinimum(1).withMaximum(1)
        .withDefault("1").create())
        .withDescription("number of threads to update the model with")
        .withShortName("nut").create();

    Option verboseOpt = obuilder.withLongName("verbose").withRequired(false)
        .withArgument(abuilder.withName("verbose").withMinimum(1).withMaximum(1)
        .withDefault("false").create())
        .withDescription("print verbose information, like top-terms in each topic, during iteration")
        .withShortName("v").create();

    Group group = gbuilder.withName("Options").withOption(inputDirOpt).withOption(numTopicsOpt)
        .withOption(alphaOpt).withOption(etaOpt)
        .withOption(maxIterOpt).withOption(burnInOpt).withOption(convergenceOpt)
        .withOption(dictOpt).withOption(reInferDocTopicsOpt)
        .withOption(outputDocFileOpt).withOption(outputTopicFileOpt).withOption(dfsOpt)
        .withOption(numTrainThreadsOpt).withOption(numUpdateThreadsOpt)
        .withOption(modelCorpusFractionOption).withOption(verboseOpt).create();

    try {
      Parser parser = new Parser();

      parser.setGroup(group);
      parser.setHelpOption(helpOpt);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return -1;
      }

      String inputDirString = (String) cmdLine.getValue(inputDirOpt);
      String dictDirString = cmdLine.hasOption(dictOpt) ? (String)cmdLine.getValue(dictOpt) : null;
      int numTopics = Integer.parseInt((String) cmdLine.getValue(numTopicsOpt));
      double alpha = Double.parseDouble((String)cmdLine.getValue(alphaOpt));
      double eta = Double.parseDouble((String)cmdLine.getValue(etaOpt));
      int maxIterations = Integer.parseInt((String)cmdLine.getValue(maxIterOpt));
      int burnInIterations = Integer.parseInt((String)cmdLine.getValue(burnInOpt));
      double minFractionalErrorChange = Double.parseDouble((String) cmdLine.getValue(convergenceOpt));
      int numTrainThreads = Integer.parseInt((String)cmdLine.getValue(numTrainThreadsOpt));
      int numUpdateThreads = Integer.parseInt((String)cmdLine.getValue(numUpdateThreadsOpt));
      String topicOutFile = (String)cmdLine.getValue(outputTopicFileOpt);
      String docOutFile = (String)cmdLine.getValue(outputDocFileOpt);
      //String reInferDocTopics = (String)cmdLine.getValue(reInferDocTopicsOpt);
      boolean verbose = Boolean.parseBoolean((String) cmdLine.getValue(verboseOpt));
      double modelCorpusFraction = Double.parseDouble((String)cmdLine.getValue(modelCorpusFractionOption));

      long start = System.nanoTime();

      if (conf.get("fs.default.name") == null) {
        String dfsNameNode = (String)cmdLine.getValue(dfsOpt);
        conf.set("fs.default.name", dfsNameNode);
      }
      String[] terms = loadDictionary(dictDirString, conf);
      logTime("dictionary loading", System.nanoTime() - start);
      start = System.nanoTime();
      Matrix corpus = loadVectors(inputDirString, conf);
      logTime("vector seqfile corpus loading", System.nanoTime() - start);
      start = System.nanoTime();
      InMemoryCollapsedVariationalBayes0 cvb0 =
          new InMemoryCollapsedVariationalBayes0(corpus, terms, numTopics, alpha, eta,
                                                 numTrainThreads, numUpdateThreads, modelCorpusFraction);
      logTime("cvb0 init", System.nanoTime() - start);

      start = System.nanoTime();
      cvb0.setVerbose(verbose);
      cvb0.iterateUntilConvergence(minFractionalErrorChange, maxIterations, burnInIterations);
      logTime("total training time", System.nanoTime() - start);

      /*
      if ("randstart".equalsIgnoreCase(reInferDocTopics)) {
        cvb0.inferDocuments(0.0, 100, true);
      } else if ("continue".equalsIgnoreCase(reInferDocTopics)) {
        cvb0.inferDocuments(0.0, 100, false);
      }
       */

      start = System.nanoTime();
      cvb0.writeModel(new Path(topicOutFile));
      DistributedRowMatrixWriter.write(new Path(docOutFile), conf, cvb0.docTopicCounts);
      logTime("printTopics", System.nanoTime() - start);
    } catch (OptionException e) {
      log.error("Error while parsing options", e);
      CommandLineUtil.printHelp(group);
    }
    return 0;
  }

  /*
  private static Map<Integer, Map<String, Integer>> loadCorpus(String path) throws IOException {
    List<String> lines = Resources.readLines(Resources.getResource(path), Charsets.UTF_8);
    Map<Integer, Map<String, Integer>> corpus = Maps.newHashMap();
    for (int i=0; i<lines.size(); i++) {
      String line = lines.get(i);
      Map<String, Integer> doc = Maps.newHashMap();
      for (String s : line.split(" ")) {
        s = s.replaceAll("\\W", "").toLowerCase().trim();
        if (s.length() == 0) {
          continue;
        }
        if (!doc.containsKey(s)) {
          doc.put(s, 0);
        }
        doc.put(s, doc.get(s) + 1);
      }
      corpus.put(i, doc);
    }
    return corpus;
  }
   */

  private static String[] loadDictionary(String dictionaryPath, Configuration conf) {
    if (dictionaryPath == null) {
      return null;
    }
    Path dictionaryFile = new Path(dictionaryPath);
    List<Pair<Integer, String>> termList = Lists.newArrayList();
    int maxTermId = 0;
     // key is word value is id
    for (Pair<Writable, IntWritable> record
            : new SequenceFileIterable<Writable, IntWritable>(dictionaryFile, true, conf)) {
      termList.add(new Pair<Integer, String>(record.getSecond().get(),
          record.getFirst().toString()));
      maxTermId = Math.max(maxTermId, record.getSecond().get());
    }
    String[] terms = new String[maxTermId + 1];
    for (Pair<Integer, String> pair : termList) {
      terms[pair.getFirst()] = pair.getSecond();
    }
    return terms;
  }

  @Override
  public Configuration getConf() {
    return super.getConf();
  }

  private static Matrix loadVectors(String vectorPathString, Configuration conf)
    throws IOException {
    Path vectorPath = new Path(vectorPathString);
    FileSystem fs = vectorPath.getFileSystem(conf);
    List<Path> subPaths = Lists.newArrayList();
    if (fs.isFile(vectorPath)) {
      subPaths.add(vectorPath);
    } else {
      for (FileStatus fileStatus : fs.listStatus(vectorPath, PathFilters.logsCRCFilter())) {
        subPaths.add(fileStatus.getPath());
      }
    }
    List<Pair<Integer, Vector>> rowList = Lists.newArrayList();
    int numRows = Integer.MIN_VALUE;
    int numCols = -1;
    boolean sequentialAccess = false;
    for (Path subPath : subPaths) {
      for (Pair<IntWritable, VectorWritable> record
          : new SequenceFileIterable<IntWritable, VectorWritable>(subPath, true, conf)) {
        int id = record.getFirst().get();
        Vector vector = record.getSecond().get();
        if (vector instanceof NamedVector) {
          vector = ((NamedVector)vector).getDelegate();
        }
        if (numCols < 0) {
          numCols = vector.size();
          sequentialAccess = vector.isSequentialAccess();
        }
        rowList.add(Pair.of(id, vector));
        numRows = Math.max(numRows, id);
      }
    }
    numRows++;
    Vector[] rowVectors = new Vector[numRows];
    for (Pair<Integer, Vector> pair : rowList) {
      rowVectors[pair.getFirst()] = pair.getSecond();
    }
    return new SparseRowMatrix(numRows, numCols, rowVectors, true, !sequentialAccess);

  }

  @Override
  public int run(String[] strings) throws Exception {
    return main2(strings, getConf());
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new InMemoryCollapsedVariationalBayes0(), args);
  }
}
