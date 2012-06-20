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

package org.apache.mahout.classifier.df.tools;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collection;
import java.util.List;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.node.Node;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This tool is to visualize the Decision Forest
 */
public final class ForestVisualizer {

  private static final Logger log = LoggerFactory.getLogger(ForestVisualizer.class);

  private ForestVisualizer() {
  }

  public static String toString(DecisionForest forest, Dataset dataset, String[] attrNames) {

    List<Node> trees;
    try {
      Method getTrees = forest.getClass().getDeclaredMethod("getTrees");
      getTrees.setAccessible(true);
      trees = (List<Node>) getTrees.invoke(forest);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InvocationTargetException e) {
      throw new IllegalStateException(e);
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(e);
    }

    int cnt = 1;
    StringBuilder buff = new StringBuilder();
    for (Node tree : trees) {
      buff.append("Tree[").append(cnt).append("]:");
      buff.append(TreeVisualizer.toString(tree, dataset, attrNames));
      buff.append('\n');
      cnt++;
    }
    return buff.toString();
  }

  /**
   * Decision Forest to String
   * @param forestPath
   *          path to the Decision Forest
   * @param datasetPath
   *          dataset path
   * @param attrNames
   *          attribute names
   */
  public static String toString(String forestPath, String datasetPath, String[] attrNames) throws IOException {
    Configuration conf = new Configuration();
    DecisionForest forest = DecisionForest.load(conf, new Path(forestPath));
    Dataset dataset = Dataset.load(conf, new Path(datasetPath));
    return toString(forest, dataset, attrNames);
  }

  /**
   * Print Decision Forest
   * @param forestPath
   *          path to the Decision Forest
   * @param datasetPath
   *          dataset path
   * @param attrNames
   *          attribute names
   */
  public static void print(String forestPath, String datasetPath, String[] attrNames) throws IOException {
    System.out.println(toString(forestPath, datasetPath, attrNames));
  }
  
  public static void main(String[] args) {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true)
      .withArgument(abuilder.withName("dataset").withMinimum(1).withMaximum(1).create())
      .withDescription("Dataset path").create();

    Option modelOpt = obuilder.withLongName("model").withShortName("m").withRequired(true)
      .withArgument(abuilder.withName("path").withMinimum(1).withMaximum(1).create())
      .withDescription("Path to the Decision Forest").create();

    Option attrNamesOpt = obuilder.withLongName("names").withShortName("n").withRequired(false)
      .withArgument(abuilder.withName("names").withMinimum(1).create())
      .withDescription("Optional, Attribute names").create();

    Option helpOpt = obuilder.withLongName("help").withShortName("h")
      .withDescription("Print out help").create();
  
    Group group = gbuilder.withName("Options").withOption(datasetOpt).withOption(modelOpt)
      .withOption(attrNamesOpt).withOption(helpOpt).create();
  
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption("help")) {
        CommandLineUtil.printHelp(group);
        return;
      }
  
      String datasetName = cmdLine.getValue(datasetOpt).toString();
      String modelName = cmdLine.getValue(modelOpt).toString();
      String[] attrNames = null;
      if (cmdLine.hasOption(attrNamesOpt)) {
        Collection<String> names = (Collection<String>) cmdLine.getValues(attrNamesOpt);
        if (!names.isEmpty()) {
          attrNames = new String[names.size()];
          names.toArray(attrNames);
        }
      }
      
      print(modelName, datasetName, attrNames);
    } catch (Exception e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }
}
