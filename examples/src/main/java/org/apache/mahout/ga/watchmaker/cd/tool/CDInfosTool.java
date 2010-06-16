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

package org.apache.mahout.ga.watchmaker.cd.tool;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.SequenceFile.Sorter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.StringUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.ga.watchmaker.OutputUtils;
import org.apache.mahout.ga.watchmaker.cd.FileInfoParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Gathers additional information about a given dataset. Takes a descriptor
 * about the attributes, and generates a description for each one.
 */
public final class CDInfosTool {

  private static final Logger log = LoggerFactory.getLogger(CDInfosTool.class);

  private CDInfosTool() {
  }

  /**
   * Uses Mahout to gather the information about a dataset.
   * 
   * @param descriptors about the available attributes
   * @param inpath input path (the dataset)
   * @param descriptions <code>List&lt;String&gt;</code> that contains the
   *        generated descriptions for each non ignored attribute
   * @throws IOException
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  public static void gatherInfos(Descriptors descriptors, Path inpath, Path output, List<String> descriptions) throws IOException,
      InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    Job job = new Job(conf);
    FileSystem fs = FileSystem.get(inpath.toUri(), conf);

    // check the input
    if (!fs.exists(inpath) || !fs.getFileStatus(inpath).isDir()) {
      throw new IllegalArgumentException("Input path not found or is not a directory");
    }

    configureJob(job, descriptors, inpath, output);
    job.waitForCompletion(true);

    importDescriptions(fs, conf, output, descriptions);
  }

  /**
   * Configure the job
   * 
   * @param job
   * @param descriptors attributes's descriptors
   * @param inpath input <code>Path</code>
   * @param outpath output <code>Path</code>
   * @throws IOException 
   */
  private static void configureJob(Job job, Descriptors descriptors, Path inpath, Path outpath) throws IOException {
    FileInputFormat.setInputPaths(job, inpath);
    FileOutputFormat.setOutputPath(job, outpath);

    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(Text.class);

    job.setMapperClass(ToolMapper.class);
    job.setCombinerClass(ToolCombiner.class);
    job.setReducerClass(ToolReducer.class);

    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    // store the stringified descriptors
    job.getConfiguration().set(ToolMapper.ATTRIBUTES, StringUtils.toString(descriptors.getChars()));
  }

  /**
   * Reads back the descriptions.
   * 
   * @param fs file system
   * @param conf job configuration
   * @param outpath output <code>Path</code>
   * @param descriptions List of attribute's descriptions
   * @throws IOException
   */
  private static void importDescriptions(FileSystem fs, Configuration conf, Path outpath, List<String> descriptions)
      throws IOException {
    Sorter sorter = new Sorter(fs, LongWritable.class, Text.class, conf);

    // merge and sort the outputs
    Path[] outfiles = OutputUtils.listOutputFiles(fs, outpath);
    Path output = new Path(outpath, "output.sorted");
    sorter.merge(outfiles, output);

    // import the descriptions
    LongWritable key = new LongWritable();
    Text value = new Text();
    Reader reader = new Reader(fs, output, conf);

    while (reader.next(key, value)) {
      descriptions.add(value.toString());
    }

    reader.close();
  }

  /**
   * Load the dataset's attributes descriptors from an .info file
   * 
   * @param inpath dataset path
   */
  private static Descriptors loadDescriptors(FileSystem fs, Path inpath) throws IOException {
    // TODO should become part of FileInfoParser

    Path infpath = FileInfoParser.getInfoFile(fs, inpath);

    FSDataInputStream input = fs.open(infpath);
    Scanner reader = new Scanner(input);

    List<Character> descriptors = new ArrayList<Character>();

    while (reader.hasNextLine()) {
      String c = reader.nextLine();
      descriptors.add(c.toUpperCase(Locale.ENGLISH).charAt(0));
    }

    if (descriptors.isEmpty()) {
      throw new IllegalArgumentException("Infos file is empty");
    }

    char[] desc = new char[descriptors.size()];
    for (int index = 0; index < descriptors.size(); index++) {
      desc[index] = descriptors.get(index);
    }

    return new Descriptors(desc);
  }

  private static void storeDescriptions(FileSystem fs, Path inpath, Descriptors descriptors, List<String> descriptions)
      throws IOException {
    // TODO should become part of FileInfoParser

    Path infpath = FileInfoParser.getInfoFile(fs, inpath);

    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fs.create(infpath)));
    try {
      int aindex = 0;
      for (int index = 0; index < descriptors.size(); index++) {
        if (descriptors.isLabel(index)) {
          writer.write(FileInfoParser.LABEL_TOKEN + ", ");
          writer.write(descriptions.get(aindex++));
        } else if (descriptors.isNumerical(index)) {
          writer.write(FileInfoParser.NUMERICAL_TOKEN + ", ");
          writer.write(descriptions.get(aindex++));
        } else if (descriptors.isNominal(index)) {
          writer.write(FileInfoParser.NOMINAL_TOKEN + ", ");
          writer.write(descriptions.get(aindex++));
        } else {
          writer.write(FileInfoParser.IGNORED_TOKEN);
        }

        writer.newLine();
      }
    } finally {
      writer.close();
    }
  }

  public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withRequired(true).withShortName("i").withArgument(
        abuilder.withName("input").withMinimum(1).withMaximum(1).create()).withDescription("The Path for input data directory.")
        .create();

    Option helpOpt = DefaultOptionCreator.helpOption();

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(helpOpt).create();

    Parser parser = new Parser();
    parser.setGroup(group);
    try {
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }

      Path input = new Path(cmdLine.getValue(inputOpt).toString());
      Path output = new Path("output"); // TODO surely this should be configurable?

      FileSystem fs = FileSystem.get(input.toUri(), new Configuration());

      log.info("Loading Descriptors...");
      Descriptors descriptors = loadDescriptors(fs, input);

      log.info("Gathering informations...");
      List<String> descriptions = new ArrayList<String>();
      gatherInfos(descriptors, input, output, descriptions);

      log.info("Storing Descriptions...");
      storeDescriptions(fs, input, descriptors, descriptions);
    } catch (OptionException e) {
      log.error("Error while parsing options", e);
      CommandLineUtil.printHelp(group);
    }
  }
}
