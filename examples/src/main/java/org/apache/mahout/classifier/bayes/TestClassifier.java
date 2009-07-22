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

package org.apache.mahout.classifier.bayes;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.bayes.io.SequenceFileModelReader;
import org.apache.mahout.classifier.cbayes.CBayesClassifier;
import org.apache.mahout.classifier.cbayes.CBayesModel;
import org.apache.mahout.common.Classifier;
import org.apache.mahout.common.Model;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.io.FilenameFilter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TestClassifier {

  private static final Logger log = LoggerFactory.getLogger(TestClassifier.class);

  private TestClassifier() {
    // do nothing
  }

  public static void main(String[] args) throws IOException,
      OptionException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option pathOpt = obuilder.withLongName("model").withRequired(true).withArgument(
            abuilder.withName("model").withMinimum(1).withMaximum(1).create()).
            withDescription("The file system path containing the model (the output path from training)").withShortName("p").create();

    Option dirOpt = obuilder.withLongName("testDir").withRequired(true).withArgument(
            abuilder.withName("testDir").withMinimum(1).withMaximum(1).create()).
            withDescription("The directory where test documents resides in").withShortName("t").create();

    Option encodingOpt = obuilder.withLongName("encoding").withArgument(
            abuilder.withName("encoding").withMinimum(1).withMaximum(1).create()).
            withDescription("The file encoding.  Defaults to UTF-8").withShortName("e").create();

    Option analyzerOpt = obuilder.withLongName("analyzer").withArgument(
            abuilder.withName("analyzer").withMinimum(1).withMaximum(1).create()).
            withDescription("The Analyzer to use").withShortName("a").create();

    Option defaultCatOpt = obuilder.withLongName("defaultCat").withArgument(
            abuilder.withName("defaultCat").withMinimum(1).withMaximum(1).create()).
            withDescription("The default category").withShortName("d").create();

    Option gramSizeOpt = obuilder.withLongName("gramSize").withRequired(true).withArgument(
            abuilder.withName("gramSize").withMinimum(1).withMaximum(1).create()).
            withDescription("Size of the n-gram").withShortName("ng").create();
    Option verboseOutputOpt = obuilder.withLongName("verbose").withRequired(false).
            withDescription("Output which values were correctly and incorrectly classified").withShortName("v").create();
    Option typeOpt = obuilder.withLongName("classifierType").withRequired(true).withArgument(
            abuilder.withName("classifierType").withMinimum(1).withMaximum(1).create()).
            withDescription("Type of classifier: bayes|cbayes").withShortName("type").create();

    Group group = gbuilder.withName("Options").withOption(analyzerOpt).withOption(defaultCatOpt).withOption(dirOpt).withOption(encodingOpt).withOption(gramSizeOpt).withOption(pathOpt)
            .withOption(typeOpt).withOption(verboseOutputOpt).create();

    Parser parser = new Parser();
    parser.setGroup(group);
    CommandLine cmdLine = parser.parse(args);

    JobConf conf = new JobConf(TestClassifier.class);

    Map<String, Path> modelPaths = new HashMap<String, Path>();
    String modelBasePath = (String) cmdLine.getValue(pathOpt);
    modelPaths.put("sigma_j", new Path(modelBasePath + "/trainer-weights/Sigma_j/part-*"));
    modelPaths.put("sigma_k", new Path(modelBasePath + "/trainer-weights/Sigma_k/part-*"));
    modelPaths.put("sigma_kSigma_j", new Path(modelBasePath + "/trainer-weights/Sigma_kSigma_j/part-*"));
    modelPaths.put("thetaNormalizer", new Path(modelBasePath + "/trainer-thetaNormalizer/part-*"));
    modelPaths.put("weight", new Path(modelBasePath + "/trainer-tfIdf/trainer-tfIdf/part-*"));

    FileSystem fs = FileSystem.get(new Path(modelBasePath).toUri(), conf);

    log.info("Loading model from: {}", modelPaths);

    Model model;
    Classifier classifier;

    String classifierType = (String) cmdLine.getValue(typeOpt);

    if (classifierType.equalsIgnoreCase("bayes")) {
      log.info("Testing Bayes Classifier");
      model = new BayesModel();
      classifier = new BayesClassifier();
    } else if (classifierType.equalsIgnoreCase("cbayes")) {
      log.info("Testing Complementary Bayes Classifier");
      model = new CBayesModel();
      classifier = new CBayesClassifier();
    } else {
      throw new IllegalArgumentException("Unrecognized classifier type: " + classifierType);
    }

    SequenceFileModelReader.loadModel(model, fs, modelPaths, conf);

    log.info("Done loading model: # labels: {}", model.getLabels().size());

    log.info("Done generating Model");

    String defaultCat = "unknown";
    if (cmdLine.hasOption(defaultCatOpt)) {
      defaultCat = (String) cmdLine.getValue(defaultCatOpt);
    }

    String encoding = "UTF-8";
    if (cmdLine.hasOption(encodingOpt)) {
      encoding = (String) cmdLine.getValue(encodingOpt);
    }
    boolean verbose = cmdLine.hasOption(verboseOutputOpt);
    //Analyzer analyzer = null;
    //if (cmdLine.hasOption(analyzerOpt)) {
      //String className = (String) cmdLine.getValue(analyzerOpt);
      //Class clazz = Class.forName(className);
      //analyzer = (Analyzer) clazz.newInstance();
    //}
    //if (analyzer == null) {
    //  analyzer = new StandardAnalyzer();
    //}
    int gramSize = 1;
    if (cmdLine.hasOption(gramSizeOpt)) {
      gramSize = Integer.parseInt((String) cmdLine
          .getValue(gramSizeOpt));

    }

    String testDirPath = (String) cmdLine.getValue(dirOpt);
    File dir = new File(testDirPath);
    File[] subdirs = dir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String s) {
        return s.startsWith(".") == false;
      }
    });

    ResultAnalyzer resultAnalyzer = new ResultAnalyzer(model.getLabels(), defaultCat);

    if (subdirs != null) {
      for (File file : subdirs) {
        log.info("--------------");
        log.info("Testing: " + file);
        String correctLabel = file.getName().split(".txt")[0];
        BufferedReader fileReader = new BufferedReader(new InputStreamReader(
            new FileInputStream(file.getPath()), encoding));
        try {
          String line;
          long lineNum = 0;
          while ((line = fileReader.readLine()) != null) {
  
            Map<String, List<String>> document = Model.generateNGrams(line, gramSize);
            for (Map.Entry<String, List<String>> stringListEntry : document.entrySet()) {
              List<String> strings = stringListEntry.getValue();
              ClassifierResult classifiedLabel = classifier.classify(model,
                  strings.toArray(new String[strings.size()]),
                  defaultCat);
              boolean correct = resultAnalyzer.addInstance(correctLabel, classifiedLabel);
              if (verbose == true){
                //We have one document per line
                log.info("Line Number: " + lineNum + " Line(30): " + (line.length() > 30 ? line.substring(0, 30) : line) +
                        " Expected Label: " + correctLabel + " Classified Label: " + classifiedLabel.getLabel() + " Correct: " + correct);
              }
            }
            lineNum++;
          }
          log.info("{}\t{}\t{}/{}", new Object[]{
              correctLabel,
              resultAnalyzer.getConfusionMatrix().getAccuracy(correctLabel),
              resultAnalyzer.getConfusionMatrix().getCorrect(correctLabel),
              resultAnalyzer.getConfusionMatrix().getTotal(correctLabel)
          });
        } finally {
          fileReader.close();
        }
      }

    }
    log.info(resultAnalyzer.summarize());

  }
}
