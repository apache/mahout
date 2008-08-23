package org.apache.mahout.classifier.bayes;

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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.PosixParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.classifier.ClassifierResult;
import org.apache.mahout.classifier.ResultAnalyzer;
import org.apache.mahout.classifier.bayes.io.SequenceFileModelReader;
import org.apache.mahout.classifier.cbayes.CBayesClassifier;
import org.apache.mahout.classifier.cbayes.CBayesModel;
import org.apache.mahout.common.Classifier;
import org.apache.mahout.common.Model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TestClassifier {

  @SuppressWarnings({ "static-access", "unchecked" })
  public static void main(String[] args) throws IOException,
      ClassNotFoundException, IllegalAccessException, InstantiationException {
    Options options = new Options();
    Option pathOpt = OptionBuilder.withLongOpt("path").isRequired().hasArg()
        .withDescription("The local file system path").create("p");
    options.addOption(pathOpt);
    Option dirOpt = OptionBuilder.withLongOpt("testDir").isRequired().hasArg()
        .withDescription("The directory where test documents resides in").create("t");
    options.addOption(dirOpt);
    Option encodingOpt = OptionBuilder.withLongOpt("encoding").hasArg()
        .withDescription("The file encoding.  defaults to UTF-8").create("e");
    options.addOption(encodingOpt);
    Option analyzerOpt = OptionBuilder.withLongOpt("analyzer").hasArg()
        .withDescription("The Analyzer to use").create("a");
    options.addOption(analyzerOpt);
    Option defaultCatOpt = OptionBuilder.withLongOpt("defaultCat").hasArg()
        .withDescription("The default category").create("d");
    options.addOption(defaultCatOpt);
    Option gramSizeOpt = OptionBuilder.withLongOpt("gramSize").hasArg()
        .withDescription("Size of the n-gram").create("ng");
    options.addOption(gramSizeOpt);
    Option typeOpt = OptionBuilder.withLongOpt("classifierType").isRequired()
        .hasArg().withDescription("Type of classifier").create("type");
    options.addOption(typeOpt);

    CommandLine cmdLine;
    try {
      PosixParser parser = new PosixParser();
      cmdLine = parser.parse(options, args);
      SequenceFileModelReader reader = new SequenceFileModelReader();
      JobConf conf = new JobConf(TestClassifier.class);

      Map<String, Path> modelPaths = new HashMap<String, Path>();
      String modelBasePath = cmdLine.getOptionValue(pathOpt.getOpt());
      modelPaths.put("sigma_j", new Path(modelBasePath + "/trainer-weights/Sigma_j/part-*"));
      modelPaths.put("sigma_k", new Path(modelBasePath + "/trainer-weights/Sigma_k/part-*"));
      modelPaths.put("sigma_kSigma_j", new Path(modelBasePath + "/trainer-weights/Sigma_kSigma_j/part-*"));
      modelPaths.put("thetaNormalizer", new Path(modelBasePath + "/trainer-thetaNormalizer/part-*"));
      modelPaths.put("weight", new Path(modelBasePath + "/trainer-tfIdf/trainer-tfIdf/part-*"));

      FileSystem fs = FileSystem.get(conf);

      System.out.println("Loading model from: " + modelPaths);

      Model model = null;
      Classifier classifier = null;
      
      String classifierType = cmdLine.getOptionValue(typeOpt.getOpt());
      
      if (classifierType.equalsIgnoreCase("bayes")) {
        System.out.println("Testing Bayes Classifier");
        model = new BayesModel();
        classifier = new BayesClassifier();
      } else if (classifierType.equalsIgnoreCase("cbayes")) {
        System.out.println("Testing Complementary Bayes Classifier");
        model = new CBayesModel();
        classifier = new CBayesClassifier();
      }
     
      model = reader.loadModel(model, fs, modelPaths, conf);

      System.out.println("Done loading model: # labels: "
          + model.getLabels().size());

      System.out.println("Done generating Model ");

     

      String defaultCat = "unknown";
      if (cmdLine.hasOption(defaultCatOpt.getOpt())) {
        defaultCat = cmdLine.getOptionValue(defaultCatOpt.getOpt());
      }

      String encoding = "UTF-8";
      if (cmdLine.hasOption(encodingOpt.getOpt())) {
        encoding = cmdLine.getOptionValue(encodingOpt.getOpt());
      }
      Analyzer analyzer = null;
      if (cmdLine.hasOption(analyzerOpt.getOpt())) {
        String className = cmdLine.getOptionValue(analyzerOpt.getOpt());
        Class clazz = Class.forName(className);
        analyzer = (Analyzer) clazz.newInstance();
      }
      if (analyzer == null) {
        analyzer = new StandardAnalyzer();
      }
      int gramSize = 1;
      if (cmdLine.hasOption(gramSizeOpt.getOpt())) {
        gramSize = Integer.parseInt(cmdLine
            .getOptionValue(gramSizeOpt.getOpt()));

      }

      String testDirPath = cmdLine.getOptionValue(dirOpt.getOpt());
      File dir = new File(testDirPath);
      File[] subdirs = dir.listFiles();

      ResultAnalyzer resultAnalyzer = new ResultAnalyzer(model.getLabels());

      if (subdirs != null) {
        for (int loop = 0; loop < subdirs.length; loop++) {

          String correctLabel = subdirs[loop].getName().split(".txt")[0];
          System.out.print(correctLabel);
          BufferedReader fileReader = new BufferedReader(new InputStreamReader(
              new FileInputStream(subdirs[loop].getPath()), encoding));
          String line;
          while ((line = fileReader.readLine()) != null) {
            
            Map<String, List<String>> document = Model.generateNGrams(line, gramSize);
            for (String labelName : document.keySet()) {
              List<String> strings = document.get(labelName);
              ClassifierResult classifiedLabel = classifier.classify(model,
                  strings.toArray(new String[strings.size()]),
                  defaultCat);
              resultAnalyzer.addInstance(correctLabel, classifiedLabel);
            }
          }
          System.out.println("\t"
              + resultAnalyzer.getConfusionMatrix().getAccuracy(correctLabel)
              + "\t"
              + resultAnalyzer.getConfusionMatrix().getCorrect(correctLabel)
              + "/"
              + resultAnalyzer.getConfusionMatrix().getTotal(correctLabel));

        }

      }
      System.out.println(resultAnalyzer.summarize());

    } catch (Exception exp) {
      exp.printStackTrace(System.err);
    }
  }
}
