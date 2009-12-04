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

package org.apache.mahout.cf.taste.hadoop.slopeone;

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
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityMapper;
import org.apache.hadoop.util.StringUtils;
import org.apache.mahout.cf.taste.hadoop.ItemItemWritable;
import org.apache.mahout.cf.taste.hadoop.ItemPrefWritable;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public final class SlopeOneAverageDiffsJob {

  /** Logger for this class. */
  private static final Logger log = LoggerFactory.getLogger(SlopeOneAverageDiffsJob.class);

  private SlopeOneAverageDiffsJob() {
  }

  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withRequired(true).withShortName("i")
      .withArgument(abuilder.withName("input").withMinimum(1).withMaximum(1).create())
      .withDescription("The Path for input preferences file.").create();
    Option outputOpt = DefaultOptionCreator.outputOption(obuilder, abuilder).create();
    Option helpOpt = DefaultOptionCreator.helpOption(obuilder);

    Option jarFileOpt = obuilder.withLongName("jarFile").withRequired(true)
      .withShortName("m").withArgument(abuilder.withName("jarFile").withMinimum(1)
      .withMaximum(1).create()).withDescription("Implementation jar.").create();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt)
        .withOption(jarFileOpt).withOption(helpOpt).create();


    CommandLine cmdLine;
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      cmdLine = parser.parse(args);
    } catch (OptionException e) {
      log.error(e.getMessage());
      CommandLineUtil.printHelp(group);
      return;
    }

    if (cmdLine.hasOption(helpOpt)) {
      CommandLineUtil.printHelp(group);
      return;
    }

    String prefsFile = cmdLine.getValue(inputOpt).toString();
    String outputPath = cmdLine.getValue(outputOpt).toString();
    String averagesOutputPath = outputPath + "/averages";
    String jarFile = cmdLine.getValue(jarFileOpt).toString();

    JobConf prefsToDiffsJobConf = buildPrefsToDiffsJobConf(prefsFile, averagesOutputPath, jarFile);
    JobClient.runJob(prefsToDiffsJobConf);

    JobConf diffsToAveragesJobConf = buildDiffsToAveragesJobConf(averagesOutputPath, outputPath, jarFile);
    JobClient.runJob(diffsToAveragesJobConf);
  }

  private static JobConf buildPrefsToDiffsJobConf(String inputPath,
                                                  String outputPath,
                                                  String jarFile) throws IOException {

    JobConf jobConf = new JobConf();
    FileSystem fs = FileSystem.get(jobConf);

    Path prefsFilePath = new Path(inputPath).makeQualified(fs);
    Path outputPathPath = new Path(outputPath).makeQualified(fs);

    if (fs.exists(outputPathPath)) {
      fs.delete(outputPathPath, true);
    }

    jobConf.set("mapred.jar", jarFile);
    jobConf.setJar(jarFile);

    jobConf.setClass("mapred.input.format.class", TextInputFormat.class, InputFormat.class);
    jobConf.set("mapred.input.dir", StringUtils.escapeString(prefsFilePath.toString()));

    jobConf.setClass("mapred.mapper.class", ToItemPrefsMapper.class, Mapper.class);
    jobConf.setClass("mapred.mapoutput.key.class", LongWritable.class, Object.class);
    jobConf.setClass("mapred.mapoutput.value.class", ItemPrefWritable.class, Object.class);

    jobConf.setClass("mapred.reducer.class", SlopeOnePrefsToDiffsReducer.class, Reducer.class);
    jobConf.setClass("mapred.output.key.class", ItemItemWritable.class, Object.class);
    jobConf.setClass("mapred.output.value.class", FloatWritable.class, Object.class);

    jobConf.setClass("mapred.output.format.class", SequenceFileOutputFormat.class, OutputFormat.class);
    jobConf.set("mapred.output.dir", StringUtils.escapeString(outputPathPath.toString()));

    return jobConf;
  }

  private static JobConf buildDiffsToAveragesJobConf(String inputPath,
                                                     String outputPath,
                                                     String jarFile) throws IOException {

    JobConf jobConf = new JobConf();
    FileSystem fs = FileSystem.get(jobConf);

    Path prefsFilePath = new Path(inputPath).makeQualified(fs);
    Path outputPathPath = new Path(outputPath).makeQualified(fs);

    if (fs.exists(outputPathPath)) {
      fs.delete(outputPathPath, true);
    }

    jobConf.set("mapred.jar", jarFile);
    jobConf.setJar(jarFile);

    jobConf.setClass("mapred.input.format.class", SequenceFileInputFormat.class, InputFormat.class);
    jobConf.set("mapred.input.dir", StringUtils.escapeString(prefsFilePath.toString()));

    jobConf.setClass("mapred.mapper.class", IdentityMapper.class, Mapper.class);
    jobConf.setClass("mapred.mapoutput.key.class", ItemItemWritable.class, Object.class);
    jobConf.setClass("mapred.mapoutput.value.class", FloatWritable.class, Object.class);

    jobConf.setClass("mapred.reducer.class", SlopeOneDiffsToAveragesReducer.class, Reducer.class);
    jobConf.setClass("mapred.output.key.class", ItemItemWritable.class, Object.class);
    jobConf.setClass("mapred.output.value.class", FloatWritable.class, Object.class);

    jobConf.setClass("mapred.output.format.class", TextOutputFormat.class, OutputFormat.class);
    jobConf.set("mapred.output.dir", StringUtils.escapeString(outputPathPath.toString()));

    return jobConf;
  }

}