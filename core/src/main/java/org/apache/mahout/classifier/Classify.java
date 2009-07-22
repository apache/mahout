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

package org.apache.mahout.classifier;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.bayes.BayesClassifier;
import org.apache.mahout.classifier.bayes.BayesModel;
import org.apache.mahout.classifier.bayes.io.SequenceFileModelReader;
import org.apache.mahout.classifier.cbayes.CBayesModel;
import org.apache.mahout.common.Classifier;
import org.apache.mahout.common.Model;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Classify {

  private static final Logger log = LoggerFactory.getLogger(Classify.class);

  private Classify() {
  }

  public static void main(String[] args)
      throws IOException, ClassNotFoundException, IllegalAccessException, InstantiationException, OptionException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option pathOpt = obuilder.withLongName("path").withRequired(true).withArgument(
        abuilder.withName("path").withMinimum(1).withMaximum(1).create()).withDescription("The local file system path").withShortName("p").create();

    Option classifyOpt = obuilder.withLongName("classify").withRequired(true).withArgument(
        abuilder.withName("classify").withMinimum(1).withMaximum(1).create()).
        withDescription("The doc to classify").withShortName("").create();

    Option encodingOpt = obuilder.withLongName("encoding").withRequired(true).withArgument(
        abuilder.withName("encoding").withMinimum(1).withMaximum(1).create()).
        withDescription("The file encoding.  Default: UTF-8").withShortName("e").create();

    Option analyzerOpt = obuilder.withLongName("analyzer").withRequired(true).withArgument(
        abuilder.withName("analyzer").withMinimum(1).withMaximum(1).create()).
        withDescription("The Analyzer to use").withShortName("a").create();

    Option defaultCatOpt = obuilder.withLongName("defaultCat").withRequired(true).withArgument(
        abuilder.withName("defaultCat").withMinimum(1).withMaximum(1).create()).
        withDescription("The default category").withShortName("d").create();

    Option gramSizeOpt = obuilder.withLongName("gramSize").withRequired(true).withArgument(
        abuilder.withName("gramSize").withMinimum(1).withMaximum(1).create()).
        withDescription("Size of the n-gram").withShortName("ng").create();

    Option typeOpt = obuilder.withLongName("classifierType").withRequired(true).withArgument(
        abuilder.withName("classifierType").withMinimum(1).withMaximum(1).create()).
        withDescription("Type of classifier").withShortName("type").create();

    Group options = gbuilder.withName("Options").withOption(pathOpt).withOption(classifyOpt).withOption(encodingOpt).withOption(analyzerOpt).withOption(defaultCatOpt).withOption(gramSizeOpt).withOption(typeOpt).create();

    Parser parser = new Parser();
    parser.setGroup(options);
    CommandLine cmdLine = parser.parse(args);

    JobConf conf = new JobConf(Classify.class);

    Map<String, Path> modelPaths = new HashMap<String, Path>();
    String modelBasePath = (String) cmdLine.getValue(pathOpt);
    modelPaths.put("sigma_j", new Path(modelBasePath + "/trainer-weights/Sigma_j/part-*"));
    modelPaths.put("sigma_k", new Path(modelBasePath + "/trainer-weights/Sigma_k/part-*"));
    modelPaths.put("sigma_kSigma_j", new Path(modelBasePath + "/trainer-weights/Sigma_kSigma_j/part-*"));
    modelPaths.put("thetaNormalizer", new Path(modelBasePath + "/trainer-thetaNormalizer/part-*"));
    modelPaths.put("weight", new Path(modelBasePath + "/trainer-tfIdf/trainer-tfIdf/part-*"));

    FileSystem fs = FileSystem.get((new Path(modelBasePath)).toUri(), conf);

    log.info("Loading model from: {}", modelPaths);

    Model model;

    String classifierType = (String) cmdLine.getValue(typeOpt);

    if (classifierType.equalsIgnoreCase("bayes")) {
      log.info("Testing Bayes Classifier");
      model = new BayesModel();
    } else if (classifierType.equalsIgnoreCase("cbayes")) {
      log.info("Testing Complementary Bayes Classifier");
      model = new CBayesModel();
    } else {
      throw new IllegalArgumentException("Unrecognized classifier type: " + classifierType);
    }
    Classifier classifier = new BayesClassifier();

    SequenceFileModelReader.loadModel(model, fs, modelPaths, conf);

    log.info("Done loading model: # labels: {}", model.getLabels().size());

    log.info("Done generating Model");


    String defaultCat = "unknown";
    if (cmdLine.hasOption(defaultCatOpt)) {
      defaultCat = (String) cmdLine.getValue(defaultCatOpt);
    }
    File docPath = new File((String) cmdLine.getValue(classifyOpt));
    String encoding = "UTF-8";
    if (cmdLine.hasOption(encodingOpt)) {
      encoding = (String) cmdLine.getValue(encodingOpt);
    }
    Analyzer analyzer = null;
    if (cmdLine.hasOption(analyzerOpt)) {
      String className = (String) cmdLine.getValue(analyzerOpt);
      analyzer = Class.forName(className).asSubclass(Analyzer.class).newInstance();
    }
    if (analyzer == null) {
      analyzer = new StandardAnalyzer(Version.LUCENE_CURRENT);
    }

    int gramSize = 1;
    if (cmdLine.hasOption(gramSizeOpt)) {
      gramSize = Integer.parseInt((String) cmdLine
          .getValue(gramSizeOpt));

    }

    log.info("Converting input document to proper format");
    String[] document = BayesFileFormatter.readerToDocument(analyzer, new InputStreamReader(new FileInputStream(docPath), Charset.forName(encoding)));
    StringBuilder line = new StringBuilder();
    for (String token : document) {
      line.append(token).append(' ');
    }
    List<String> doc = Model.generateNGramsWithoutLabel(line.toString(), gramSize);
    log.info("Done converting");
    log.info("Classifying document: {}", docPath);
    ClassifierResult category = classifier.classify(model, doc.toArray(new String[doc.size()]), defaultCat);
    log.info("Category for {} is {}", docPath, category);

  }
}
