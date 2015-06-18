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
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.DescriptorException;
import org.apache.mahout.classifier.df.data.DescriptorUtils;
import org.apache.mahout.common.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Generates a file descriptor for a given dataset
 */
public final class Describe implements Tool {

  private static final Logger log = LoggerFactory.getLogger(Describe.class);

  private Describe() {}

  public static int main(String[] args) throws Exception {
    return ToolRunner.run(new Describe(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option pathOpt = obuilder.withLongName("path").withShortName("p").withRequired(true).withArgument(
        abuilder.withName("path").withMinimum(1).withMaximum(1).create()).withDescription("Data path").create();

    Option descriptorOpt = obuilder.withLongName("descriptor").withShortName("d").withRequired(true)
        .withArgument(abuilder.withName("descriptor").withMinimum(1).create()).withDescription(
            "data descriptor").create();

    Option descPathOpt = obuilder.withLongName("file").withShortName("f").withRequired(true).withArgument(
        abuilder.withName("file").withMinimum(1).withMaximum(1).create()).withDescription(
        "Path to generated descriptor file").create();

    Option regOpt = obuilder.withLongName("regression").withDescription("Regression Problem").withShortName("r")
        .create();

    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();

    Group group = gbuilder.withName("Options").withOption(pathOpt).withOption(descPathOpt).withOption(
        descriptorOpt).withOption(regOpt).withOption(helpOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return -1;
      }

      String dataPath = cmdLine.getValue(pathOpt).toString();
      String descPath = cmdLine.getValue(descPathOpt).toString();
      List<String> descriptor = convert(cmdLine.getValues(descriptorOpt));
      boolean regression = cmdLine.hasOption(regOpt);

      log.debug("Data path : {}", dataPath);
      log.debug("Descriptor path : {}", descPath);
      log.debug("Descriptor : {}", descriptor);
      log.debug("Regression : {}", regression);

      runTool(dataPath, descriptor, descPath, regression);
    } catch (OptionException e) {
      log.warn(e.toString());
      CommandLineUtil.printHelp(group);
    }
    return 0;
  }

  private void runTool(String dataPath, Iterable<String> description, String filePath, boolean regression)
    throws DescriptorException, IOException {
    log.info("Generating the descriptor...");
    String descriptor = DescriptorUtils.generateDescriptor(description);

    Path fPath = validateOutput(filePath);

    log.info("generating the dataset...");
    Dataset dataset = generateDataset(descriptor, dataPath, regression);

    log.info("storing the dataset description");
    String json = dataset.toJSON();
    DFUtils.storeString(conf, fPath, json);
  }

  private Dataset generateDataset(String descriptor, String dataPath, boolean regression) throws IOException,
      DescriptorException {
    Path path = new Path(dataPath);
    FileSystem fs = path.getFileSystem(conf);

    return DataLoader.generateDataset(descriptor, regression, fs, path);
  }

  private Path validateOutput(String filePath) throws IOException {
    Path path = new Path(filePath);
    FileSystem fs = path.getFileSystem(conf);
    if (fs.exists(path)) {
      throw new IllegalStateException("Descriptor's file already exists");
    }

    return path;
  }

  private static List<String> convert(Collection<?> values) {
    List<String> list = new ArrayList<>(values.size());
    for (Object value : values) {
      list.add(value.toString());
    }
    return list;
  }

  private Configuration conf;

  @Override
  public void setConf(Configuration entries) {
    this.conf = entries;
  }

  @Override
  public Configuration getConf() {
    return conf;
  }
}
