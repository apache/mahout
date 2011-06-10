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
package org.apache.mahout.classifier.bayes.mapreduce.common;

import java.io.IOException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.bayes.BayesParameters;
import org.apache.mahout.classifier.bayes.mapreduce.bayes.BayesDriver;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base class for executing the Bayes Map/Reduce Jobs
 * 
 */
public final class JobExecutor {
  /** Logger for this class. */
  private static final Logger log = LoggerFactory.getLogger(BayesDriver.class);
  
  private JobExecutor() { }
  
  /**
   * Execute a bayes classification job. Input and output path are parsed from the input parameters.
   * 
   * @param args
   *          input parameters.
   * @param job
   *          the job to execute.
   * @throws Exception
   *           any exception thrown at job execution.
   * */
  public static void execute(String[] args, BayesJob job) throws IOException {
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputOpt = DefaultOptionCreator.inputOption().create();
    Option outputOpt = DefaultOptionCreator.outputOption().create();
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(helpOpt)
        .create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      Path input = new Path(cmdLine.getValue(inputOpt).toString());
      Path output = new Path(cmdLine.getValue(outputOpt).toString());

      BayesParameters bayesParams = new BayesParameters();
      bayesParams.setGramSize(1);
      job.runJob(input, output, bayesParams);
    } catch (OptionException e) {
      log.error(e.getMessage());
      CommandLineUtil.printHelp(group);
    }
  }
}
