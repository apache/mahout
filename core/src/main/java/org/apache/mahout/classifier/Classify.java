package org.apache.mahout.classifier;
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
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.classifier.bayes.BayesClassifier;
import org.apache.mahout.classifier.bayes.BayesModel;
import org.apache.mahout.classifier.bayes.io.SequenceFileModelReader;
import org.apache.mahout.classifier.cbayes.CBayesClassifier;
import org.apache.mahout.classifier.cbayes.CBayesModel;
import org.apache.mahout.common.Classifier;
import org.apache.mahout.common.Model;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Classify {

  private static final Logger log = LoggerFactory.getLogger(Classify.class);

  @SuppressWarnings({ "static-access" })
  public static void main(String[] args)
      throws IOException, ClassNotFoundException, IllegalAccessException, InstantiationException, ParseException {
    Options options = new Options();
    Option pathOpt = OptionBuilder.withLongOpt("path").isRequired().hasArg().withDescription("The local file system path").create("p");
    options.addOption(pathOpt);
    Option classifyOpt = OptionBuilder.withLongOpt("classify").isRequired().hasArg().withDescription("The document to classify").create("c");
    options.addOption(classifyOpt);
    Option encodingOpt = OptionBuilder.withLongOpt("encoding").hasArg().withDescription("The file encoding.  defaults to UTF-8").create("e");
    options.addOption(encodingOpt);
    Option analyzerOpt = OptionBuilder.withLongOpt("analyzer").hasArg().withDescription("The Analyzer to use").create("a");
    options.addOption(analyzerOpt);
    Option defaultCatOpt = OptionBuilder.withLongOpt("defaultCat").hasArg().withDescription("The default category").create("d");
    options.addOption(defaultCatOpt);
    Option gramSizeOpt = OptionBuilder.withLongOpt("gramSize").hasArg().withDescription("Size of the n-gram").create("ng");
    options.addOption(gramSizeOpt);
    Option typeOpt = OptionBuilder.withLongOpt("classifierType").isRequired().hasArg().withDescription("Type of classifier").create("type");
    options.addOption(typeOpt);

    PosixParser parser = new PosixParser();
    CommandLine cmdLine = parser.parse(options, args);
    SequenceFileModelReader reader = new SequenceFileModelReader();
    JobConf conf = new JobConf(Classify.class);

    Map<String, Path> modelPaths = new HashMap<String, Path>();
    String modelBasePath = cmdLine.getOptionValue(pathOpt.getOpt());
    modelPaths.put("sigma_j", new Path(modelBasePath + "/trainer-weights/Sigma_j/part-*"));
    modelPaths.put("sigma_k", new Path(modelBasePath + "/trainer-weights/Sigma_k/part-*"));
    modelPaths.put("sigma_kSigma_j", new Path(modelBasePath + "/trainer-weights/Sigma_kSigma_j/part-*"));
    modelPaths.put("thetaNormalizer", new Path(modelBasePath + "/trainer-thetaNormalizer/part-*"));
    modelPaths.put("weight", new Path(modelBasePath + "/trainer-tfIdf/trainer-tfIdf/part-*"));

    FileSystem fs = FileSystem.get(conf);

    log.info("Loading model from: {}", modelPaths);

    Model model = null;
    Classifier classifier = null;

    String classifierType = cmdLine.getOptionValue(typeOpt.getOpt());

    if (classifierType.equalsIgnoreCase("bayes")) {
      log.info("Testing Bayes Classifier");
      model = new BayesModel();
      classifier = new BayesClassifier();
    } else if (classifierType.equalsIgnoreCase("cbayes")) {
      log.info("Testing Complementary Bayes Classifier");
      model = new CBayesModel();
      classifier = new CBayesClassifier();
    }

    model = reader.loadModel(model, fs, modelPaths, conf);

    log.info("Done loading model: # labels: {}", model.getLabels().size());

    log.info("Done generating Model");


    String defaultCat = "unknown";
    if (cmdLine.hasOption(defaultCatOpt.getOpt())) {
      defaultCat = cmdLine.getOptionValue(defaultCatOpt.getOpt());
    }
    File docPath = new File(cmdLine.getOptionValue(classifyOpt.getOpt()));
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

    log.info("Converting input document to proper format");
    String[] document = BayesFileFormatter.readerToDocument(analyzer, new InputStreamReader(new FileInputStream(docPath), encoding));
    StringBuilder line = new StringBuilder();
    for(String token : document)
    {
      line.append(token).append(' ');
    }
    List<String> doc = Model.generateNGramsWithoutLabel(line.toString(), gramSize) ;
    log.info("Done converting");
    log.info("Classifying document: {}", docPath);
    ClassifierResult category = classifier.classify(model, doc.toArray(new String[doc.size()]), defaultCat);
    log.info("Category for {} is {}", docPath, category);

  }
}
