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
import org.apache.commons.cli.ParseException;
import org.apache.mahout.classifier.cbayes.CBayesDriver;

import java.io.IOException;

/**
 * Train the Naive Bayes Complement classifier with improved weighting on the Twenty Newsgroups data (http://people.csail.mit.edu/jrennie/20Newsgroups/20news-18828.tar.gz)
 *
 * To run:
 * Assume MAHOUT_HOME refers to the location where you checked out/installed Mahout
 * <ol>
 * <li>From the main dir: ant extract-20news-18828</li>
 * <li>ant examples-job</li>
 * <li>Start up Hadoop and copy the files to the system. See http://hadoop.apache.org/core/docs/r0.16.2/quickstart.html</li>
 * <li>From the Hadoop dir (where Hadoop is installed):
 * <ol>
 *      <li>emacs conf/hadoop-site.xml (add in local settings per quickstart)</li>
 *      <li>bin/hadoop namenode -format  //Format the HDFS</li>
 *      <li>bin/start-all.sh  //Start Hadoop</li>
 *      <li>bin/hadoop dfs -put &lt;MAHOUT_HOME&gt;/work/20news-18828-collapse 20newsInput  //Copies the extracted text to HDFS</li>
 *      <li>bin/hadoop jar &lt;MAHOUT_HOME&gt;/build/apache-mahout-0.1-dev-ex.jar org.apache.mahout.examples.classifiers.cbayes.TwentyNewsgroups -t -i 20newsInput -o 20newsOutput</li>
 * </ol>
 *  </li>
 * </ol>
 */
public class TrainClassifier {

  public void trainNaiveBayes(String dir, String outputDir, int gramSize) throws IOException {
    BayesDriver.runJob(dir, outputDir, gramSize);
  }
  
  public void trainCNaiveBayes(String dir, String outputDir, int gramSize) throws IOException {
    CBayesDriver.runJob(dir, outputDir, gramSize);
  }
  
  @SuppressWarnings("static-access")
  public static void main(String[] args) throws IOException, ParseException {
    Options options = new Options();
    Option trainOpt = OptionBuilder.withLongOpt("train").withDescription("Train the classifier").create("t");
    options.addOption(trainOpt);
    Option inputDirOpt = OptionBuilder.withLongOpt("inputDir").hasArg().withDescription("The Directory on HDFS containing the collapsed, properly formatted files").create("i");
    options.addOption(inputDirOpt);
    Option outputOpt = OptionBuilder.withLongOpt("output").isRequired().hasArg().withDescription("The location of the model on the HDFS").create("o");
    options.addOption(outputOpt);
    Option gramSizeOpt = OptionBuilder.withLongOpt("gramSize").hasArg().withDescription("Size of the n-gram").create("ng");
    options.addOption(gramSizeOpt);
    Option typeOpt = OptionBuilder.withLongOpt("classifierType").isRequired().hasArg().withDescription("Type of classifier").create("type");
    options.addOption(typeOpt);
    
    PosixParser parser = new PosixParser();
    CommandLine cmdLine = parser.parse(options, args);

    boolean train = cmdLine.hasOption(trainOpt.getOpt());
    TrainClassifier tn = new TrainClassifier();
    if (train == true){
      String classifierType = cmdLine.getOptionValue(typeOpt.getOpt());
      if(classifierType.equalsIgnoreCase("bayes")){
        System.out.println("Training Bayes Classifier");
        tn.trainNaiveBayes(cmdLine.getOptionValue(inputDirOpt.getOpt()), cmdLine.getOptionValue(outputOpt.getOpt()), Integer.parseInt(cmdLine.getOptionValue(gramSizeOpt.getOpt())));

      } else if(classifierType.equalsIgnoreCase("cbayes")) {
        System.out.println("Training Complementary Bayes Classifier");
        //setup the HDFS and copy the files there, then run the trainer
        tn.trainCNaiveBayes(cmdLine.getOptionValue(inputDirOpt.getOpt()), cmdLine.getOptionValue(outputOpt.getOpt()), Integer.parseInt(cmdLine.getOptionValue(gramSizeOpt.getOpt())));
      }
    }

  }
}
