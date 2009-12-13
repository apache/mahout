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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.commons.cli2.Argument;
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
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.Tool;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public abstract class AbstractJob implements Tool {

  private static final Logger log = LoggerFactory.getLogger(AbstractJob.class);

  private Configuration configuration;

  @Override
  public Configuration getConf() {
    return configuration;
  }

  @Override
  public void setConf(Configuration configuration) {
    this.configuration = configuration;
  }

  protected static Option buildOption(String name, String shortName, String description) {
    return buildOption(name, shortName, description, true, null);
  }

  protected static Option buildOption(String name, String shortName, String description, String defaultValue) {
    return buildOption(name, shortName, description, false, defaultValue);    
  }

  private static Option buildOption(String name,
                                    String shortName,
                                    String description,
                                    boolean required,
                                    String defaultValue) {
    ArgumentBuilder argBuilder = new ArgumentBuilder().withName(name).withMinimum(1).withMaximum(1);
    if (defaultValue != null) {
      argBuilder = argBuilder.withDefault(defaultValue);
    }
    Argument arg = argBuilder.create();
    return new DefaultOptionBuilder().withLongName(name).withRequired(required)
      .withShortName(shortName).withArgument(arg).withDescription(description).create();
  }

  protected static Map<String,String> parseArguments(String[] args, Option... extraOpts) {

    Option inputOpt = DefaultOptionCreator.inputOption().create();
    Option tempDirOpt = buildOption("tempDir", "t", "Intermediate output directory", "temp");
    Option outputOpt = DefaultOptionCreator.outputOption().create();
    Option helpOpt = DefaultOptionCreator.helpOption();
    Option jarFileOpt = buildOption("jarFile", "m", "Implementation jar");

    GroupBuilder gBuilder = new GroupBuilder().withName("Options")
      .withOption(inputOpt)
      .withOption(tempDirOpt)
      .withOption(outputOpt)
      .withOption(helpOpt)
      .withOption(jarFileOpt);

    for (Option opt : extraOpts) {
      gBuilder = gBuilder.withOption(opt);
    }

    Group group = gBuilder.create();

    CommandLine cmdLine;
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      cmdLine = parser.parse(args);
    } catch (OptionException e) {
      log.error(e.getMessage());
      CommandLineUtil.printHelp(group);
      return null;
    }

    if (cmdLine.hasOption(helpOpt)) {
      CommandLineUtil.printHelp(group);
      return null;
    }

    Map<String,String> result = new HashMap<String,String>();
    maybePut(result, cmdLine, inputOpt);
    maybePut(result, cmdLine, tempDirOpt);
    maybePut(result, cmdLine, outputOpt);
    maybePut(result, cmdLine, helpOpt);
    maybePut(result, cmdLine, jarFileOpt);
    for (Option opt : extraOpts) {
      maybePut(result, cmdLine, opt);
    }

    return result;    
  }

  private static void maybePut(Map<String,String> args, CommandLine cmdLine, Option opt) {
    Object value = cmdLine.getValue(opt);
    if (value != null) {
      args.put(opt.getPreferredName(), value.toString());
    }
  }

  protected static JobConf prepareJobConf(String inputPath,
                                          String outputPath,
                                          String jarFile,
                                          Class<? extends InputFormat> inputFormat,
                                          Class<? extends Mapper> mapper,
                                          Class<? extends Writable> mapperKey,
                                          Class<? extends Writable> mapperValue,
                                          Class<? extends Reducer> reducer,
                                          Class<? extends Writable> reducerKey,
                                          Class<? extends Writable> reducerValue,
                                          Class<? extends OutputFormat> outputFormat) throws IOException {

    JobConf jobConf = new JobConf();
    FileSystem fs = FileSystem.get(jobConf);

    Path inputPathPath = new Path(inputPath).makeQualified(fs);
    Path outputPathPath = new Path(outputPath).makeQualified(fs);

    jobConf.set("mapred.jar", jarFile);
    jobConf.setJar(jarFile);

    jobConf.setClass("mapred.input.format.class", inputFormat, InputFormat.class);
    jobConf.set("mapred.input.dir", StringUtils.escapeString(inputPathPath.toString()));

    jobConf.setClass("mapred.mapper.class", mapper, Mapper.class);
    jobConf.setClass("mapred.mapoutput.key.class", mapperKey, Writable.class);
    jobConf.setClass("mapred.mapoutput.value.class", mapperValue, Writable.class);

    jobConf.setClass("mapred.reducer.class", reducer, Reducer.class);
    jobConf.setClass("mapred.output.key.class", reducerKey, Writable.class);
    jobConf.setClass("mapred.output.value.class", reducerValue, Writable.class);
    jobConf.setBoolean("mapred.output.compress", true);

    jobConf.setClass("mapred.output.format.class", outputFormat, OutputFormat.class);
    jobConf.set("mapred.output.dir", StringUtils.escapeString(outputPathPath.toString()));

    return jobConf;
  }

}
