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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.SequenceFile.Sorter;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.mahout.ga.watchmaker.OutputUtils;
import org.apache.mahout.ga.watchmaker.cd.FileInfoParser;
import org.apache.mahout.utils.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Gathers additional information about a given dataset. Takes a descriptor
 * about the attributes, and generates a description for each one.
 */
public class CDInfosTool {

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
   */
  public static void gatherInfos(Descriptors descriptors, Path inpath,
      List<String> descriptions) throws IOException {
    JobConf conf = new JobConf(CDInfosTool.class);
    FileSystem fs = FileSystem.get(inpath.toUri(), conf);

    // check the input
    if (!fs.exists(inpath) || !fs.getFileStatus(inpath).isDir())
      throw new RuntimeException("Input path not found or is not a directory");

    Path outpath = OutputUtils.prepareOutput(fs);

    configureJob(conf, descriptors, inpath, outpath);
    JobClient.runJob(conf);

    importDescriptions(fs, conf, outpath, descriptions);
  }

  /**
   * Configure the job
   * 
   * @param conf
   * @param descriptors attributes's descriptors
   * @param inpath input <code>Path</code>
   * @param outpath output <code>Path</code>
   */
  private static void configureJob(JobConf conf, Descriptors descriptors,
      Path inpath, Path outpath) {
    FileInputFormat.setInputPaths(conf, inpath);
    FileOutputFormat.setOutputPath(conf, outpath);

    conf.setOutputKeyClass(LongWritable.class);
    conf.setOutputValueClass(Text.class);

    conf.setMapperClass(ToolMapper.class);
    conf.setCombinerClass(ToolCombiner.class);
    conf.setReducerClass(ToolReducer.class);

    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    // store the stringified descriptors
    conf.set(ToolMapper.ATTRIBUTES, StringUtils.toString(descriptors.getChars()));
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
  private static void importDescriptions(FileSystem fs, JobConf conf,
      Path outpath, List<String> descriptions) throws IOException {
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
   * @return
   * @throws IOException
   */
  private static Descriptors loadDescriptors(FileSystem fs, Path inpath)
      throws IOException {
    // TODO should become part of FileInfoParser

    Path infpath = FileInfoParser.getInfoFile(fs, inpath);

    FSDataInputStream input = fs.open(infpath);
    Scanner reader = new Scanner(input);

    List<Character> descriptors = new ArrayList<Character>();

    while (reader.hasNextLine()) {
      String c = reader.nextLine();
        descriptors.add(c.toUpperCase().charAt(0));
    }

    if (descriptors.isEmpty()) {
      throw new RuntimeException("Infos file is empty");
    }

    char[] desc = new char[descriptors.size()];
    for (int index = 0; index < descriptors.size(); index++) {
      desc[index] = descriptors.get(index);
    }

    return new Descriptors(desc);
  }

  private static void storeDescriptions(FileSystem fs, Path inpath,
      Descriptors descriptors, List<String> descriptions) throws IOException {
    // TODO should become part of FileInfoParser

    Path infpath = FileInfoParser.getInfoFile(fs, inpath);

    FSDataOutputStream out = fs.create(infpath);
    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(out));

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

    writer.close();
  }

  public static void main(String[] args) throws IOException {
    // command-line parameters
    if (args.length == 0) {
      log.warn("Usage: CDInfosTool dataset_path");
      throw new IllegalArgumentException();
    }

    Path inpath = new Path(args[0]);
    FileSystem fs = FileSystem.get(inpath.toUri(), new Configuration());

    log.info("Loading Descriptors...");
    Descriptors descriptors = loadDescriptors(fs, inpath);

    log.info("Gathering informations...");
    List<String> descriptions = new ArrayList<String>();
    gatherInfos(descriptors, inpath, descriptions);

    log.info("Storing Descriptions...");
    storeDescriptions(fs, inpath, descriptors, descriptions);
  }
}
