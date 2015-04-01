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

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;

/**
 * Compute the frequency distribution of the "class label"<br>
 * This class can be used when the criterion variable is the categorical attribute.
 */
public final class Frequencies extends Configured implements Tool {
  
  private static final Logger log = LoggerFactory.getLogger(Frequencies.class);
  
  private Frequencies() { }
  
  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option dataOpt = obuilder.withLongName("data").withShortName("d").withRequired(true).withArgument(
      abuilder.withName("path").withMinimum(1).withMaximum(1).create()).withDescription("Data path").create();
    
    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true).withArgument(
      abuilder.withName("path").withMinimum(1).create()).withDescription("dataset path").create();
    
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();
    
    Group group = gbuilder.withName("Options").withOption(dataOpt).withOption(datasetOpt).withOption(helpOpt)
        .create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return 0;
      }
      
      String dataPath = cmdLine.getValue(dataOpt).toString();
      String datasetPath = cmdLine.getValue(datasetOpt).toString();
      
      log.debug("Data path : {}", dataPath);
      log.debug("Dataset path : {}", datasetPath);
      
      runTool(dataPath, datasetPath);
    } catch (OptionException e) {
      log.warn(e.toString(), e);
      CommandLineUtil.printHelp(group);
    }
    
    return 0;
  }
  
  private void runTool(String data, String dataset) throws IOException,
                                                   ClassNotFoundException,
                                                   InterruptedException {
    
    FileSystem fs = FileSystem.get(getConf());
    Path workingDir = fs.getWorkingDirectory();
    
    Path dataPath = new Path(data);
    Path datasetPath = new Path(dataset);
    
    log.info("Computing the frequencies...");
    FrequenciesJob job = new FrequenciesJob(new Path(workingDir, "output"), dataPath, datasetPath);
    
    int[][] counts = job.run(getConf());
    
    // outputing the frequencies
    log.info("counts[partition][class]");
    for (int[] count : counts) {
      log.info(Arrays.toString(count));
    }
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new Frequencies(), args);
  }
  
}
