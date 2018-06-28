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

package org.apache.mahout.fpm.pfpgrowth;

import java.io.IOException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.fpm.pfpgrowth.dataset.KeyBasedStringTupleGrouper;

public final class DeliciousTagsExample {
  private DeliciousTagsExample() { }
  
  public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    Option inputDirOpt = DefaultOptionCreator.inputOption().create();
    
    Option outputOpt = DefaultOptionCreator.outputOption().create();
    
    Option helpOpt = DefaultOptionCreator.helpOption();
    Option recordSplitterOpt = obuilder.withLongName("splitterPattern").withArgument(
      abuilder.withName("splitterPattern").withMinimum(1).withMaximum(1).create()).withDescription(
      "Regular Expression pattern used to split given line into fields."
          + " Default value splits comma or tab separated fields."
          + " Default Value: \"[ ,\\t]*\\t[ ,\\t]*\" ").withShortName("regex").create();
    Option encodingOpt = obuilder.withLongName("encoding").withArgument(
      abuilder.withName("encoding").withMinimum(1).withMaximum(1).create()).withDescription(
      "(Optional) The file encoding.  Default value: UTF-8").withShortName("e").create();
    Group group = gbuilder.withName("Options").withOption(inputDirOpt).withOption(outputOpt).withOption(
      helpOpt).withOption(recordSplitterOpt).withOption(encodingOpt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      Parameters params = new Parameters();
      if (cmdLine.hasOption(recordSplitterOpt)) {
        params.set("splitPattern", (String) cmdLine.getValue(recordSplitterOpt));
      }
      
      String encoding = "UTF-8";
      if (cmdLine.hasOption(encodingOpt)) {
        encoding = (String) cmdLine.getValue(encodingOpt);
      }
      params.set("encoding", encoding);
      String inputDir = (String) cmdLine.getValue(inputDirOpt);
      String outputDir = (String) cmdLine.getValue(outputOpt);
      params.set("input", inputDir);
      params.set("output", outputDir);
      params.set("groupingFieldCount", "2");
      params.set("gfield0", "1");
      params.set("gfield1", "2");
      params.set("selectedFieldCount", "1");
      params.set("field0", "3");
      params.set("maxTransactionLength", "100");
      KeyBasedStringTupleGrouper.startJob(params);
      
    } catch (OptionException ex) {
      CommandLineUtil.printHelp(group);
    }
    
  }
}
