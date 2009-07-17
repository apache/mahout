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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.mahout.classifier.cbayes.CBayesDriver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Train the Naive Bayes Complement classifier with improved weighting on the Twenty Newsgroups data (http://people.csail.mit.edu/jrennie/20Newsgroups/20news-18828.tar.gz)
 * <p/>
 * To run:
 * Assume MAHOUT_HOME refers to the location where you checked out/installed Mahout
 * <ol>
 * <li>From the main dir: ant extract-20news-18828</li>
 * <li>ant job</li>
 * <li>Start up Hadoop and copy the files to the system. See http://hadoop.apache.org/core/docs/r0.16.2/quickstart.html</li>
 * <li>From the Hadoop dir (where Hadoop is installed):
 * <ol>
 * <li>emacs conf/hadoop-site.xml (add in local settings per quickstart)</li>
 * <li>bin/hadoop namenode -format  //Format the HDFS</li>
 * <li>bin/start-all.sh  //Start Hadoop</li>
 * <li>bin/hadoop dfs -put &lt;MAHOUT_HOME&gt;/work/20news-18828-collapse 20newsInput  //Copies the extracted text to HDFS</li>
 * <li>bin/hadoop jar &lt;MAHOUT_HOME&gt;/build/apache-mahout-0.1-dev-ex.jar org.apache.mahout.classifier.bayes.TraingClassifier -t -i 20newsInput -o 20newsOutput</li>
 * </ol>
 * </li>
 * </ol>
 */
public class TrainClassifier {

  private static final Logger log = LoggerFactory.getLogger(TrainClassifier.class);

  private TrainClassifier() {
  }

  public static void trainNaiveBayes(String dir, String outputDir, int gramSize) throws IOException {
    BayesDriver.runJob(dir, outputDir, gramSize);
  }
  
  public static void trainCNaiveBayes(String dir, String outputDir, int gramSize) throws IOException {
    CBayesDriver.runJob(dir, outputDir, gramSize);
  }

  public static void main(String[] args) throws IOException, OptionException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputDirOpt = obuilder.withLongName("input").withRequired(true).withArgument(
            abuilder.withName("input").withMinimum(1).withMaximum(1).create()).
            withDescription("The Directory on HDFS containing the collapsed, properly formatted files").withShortName("i").create();

    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
            withDescription("The location of the modelon the HDFS").withShortName("o").create();

    Option gramSizeOpt = obuilder.withLongName("gramSize").withRequired(true).withArgument(
            abuilder.withName("gramSize").withMinimum(1).withMaximum(1).create()).
            withDescription("Size of the n-gram").withShortName("ng").create();

    Option typeOpt = obuilder.withLongName("classifierType").withRequired(true).withArgument(
            abuilder.withName("classifierType").withMinimum(1).withMaximum(1).create()).
            withDescription("Type of classifier: bayes or cbayes").withShortName("type").create();
    Group group = gbuilder.withName("Options").withOption(gramSizeOpt).withOption(inputDirOpt).withOption(outputOpt).withOption(typeOpt).create();
    Parser parser = new Parser();
    parser.setGroup(group);
    CommandLine cmdLine = parser.parse(args);
    String classifierType = (String) cmdLine.getValue(typeOpt);
    if (classifierType.equalsIgnoreCase("bayes")) {
      log.info("Training Bayes Classifier");
      trainNaiveBayes((String)cmdLine.getValue(inputDirOpt), (String)cmdLine.getValue(outputOpt), Integer.parseInt((String) cmdLine.getValue(gramSizeOpt)));

    } else if (classifierType.equalsIgnoreCase("cbayes")) {
      log.info("Training Complementary Bayes Classifier");
      //setup the HDFS and copy the files there, then run the trainer
      trainCNaiveBayes((String) cmdLine.getValue(inputDirOpt), (String) cmdLine.getValue(outputOpt), Integer.parseInt((String) cmdLine.getValue(gramSizeOpt)));
    }
  }
}
