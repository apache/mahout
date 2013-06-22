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

import java.io.File;
import java.io.IOException;
import java.util.Locale;
import java.util.Random;
import java.util.Scanner;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
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
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This tool is used to uniformly distribute the class of all the tuples of the dataset over a given number of
 * partitions.<br>
 * This class can be used when the criterion variable is the categorical attribute.
 */
public final class UDistrib {
  
  private static final Logger log = LoggerFactory.getLogger(UDistrib.class);
  
  private UDistrib() {}
  
  /**
   * Launch the uniform distribution tool. Requires the following command line arguments:<br>
   * 
   * data : data path dataset : dataset path numpartitions : num partitions output : output path
   *
   * @throws java.io.IOException
   */
  public static void main(String[] args) throws IOException {
    
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option dataOpt = obuilder.withLongName("data").withShortName("d").withRequired(true).withArgument(
      abuilder.withName("data").withMinimum(1).withMaximum(1).create()).withDescription("Data path").create();
    
    Option datasetOpt = obuilder.withLongName("dataset").withShortName("ds").withRequired(true).withArgument(
      abuilder.withName("dataset").withMinimum(1).create()).withDescription("Dataset path").create();
    
    Option outputOpt = obuilder.withLongName("output").withShortName("o").withRequired(true).withArgument(
      abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
      "Path to generated files").create();
    
    Option partitionsOpt = obuilder.withLongName("numpartitions").withShortName("p").withRequired(true)
        .withArgument(abuilder.withName("numparts").withMinimum(1).withMinimum(1).create()).withDescription(
          "Number of partitions to create").create();
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();
    
    Group group = gbuilder.withName("Options").withOption(dataOpt).withOption(outputOpt).withOption(
      datasetOpt).withOption(partitionsOpt).withOption(helpOpt).create();
    
    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      String data = cmdLine.getValue(dataOpt).toString();
      String dataset = cmdLine.getValue(datasetOpt).toString();
      int numPartitions = Integer.parseInt(cmdLine.getValue(partitionsOpt).toString());
      String output = cmdLine.getValue(outputOpt).toString();
      
      runTool(data, dataset, output, numPartitions);
    } catch (OptionException e) {
      log.warn(e.toString(), e);
      CommandLineUtil.printHelp(group);
    }
    
  }
  
  private static void runTool(String dataStr, String datasetStr, String output, int numPartitions) throws IOException {

    Preconditions.checkArgument(numPartitions > 0, "numPartitions <= 0");
    
    // make sure the output file does not exist
    Path outputPath = new Path(output);
    Configuration conf = new Configuration();
    FileSystem fs = outputPath.getFileSystem(conf);

    Preconditions.checkArgument(!fs.exists(outputPath), "Output path already exists");
    
    // create a new file corresponding to each partition
    // Path workingDir = fs.getWorkingDirectory();
    // FileSystem wfs = workingDir.getFileSystem(conf);
    // File parentFile = new File(workingDir.toString());
    // File tempFile = FileUtil.createLocalTempFile(parentFile, "Parts", true);
    // File tempFile = File.createTempFile("df.tools.UDistrib","");
    // tempFile.deleteOnExit();
    File tempFile = FileUtil.createLocalTempFile(new File(""), "df.tools.UDistrib", true);
    Path partsPath = new Path(tempFile.toString());
    FileSystem pfs = partsPath.getFileSystem(conf);
    
    Path[] partPaths = new Path[numPartitions];
    FSDataOutputStream[] files = new FSDataOutputStream[numPartitions];
    for (int p = 0; p < numPartitions; p++) {
      partPaths[p] = new Path(partsPath, String.format(Locale.ENGLISH, "part.%03d", p));
      files[p] = pfs.create(partPaths[p]);
    }
    
    Path datasetPath = new Path(datasetStr);
    Dataset dataset = Dataset.load(conf, datasetPath);
    
    // currents[label] = next partition file where to place the tuple
    int[] currents = new int[dataset.nblabels()];
    
    // currents is initialized randomly in the range [0, numpartitions[
    Random random = RandomUtils.getRandom();
    for (int c = 0; c < currents.length; c++) {
      currents[c] = random.nextInt(numPartitions);
    }
    
    // foreach tuple of the data
    Path dataPath = new Path(dataStr);
    FileSystem ifs = dataPath.getFileSystem(conf);
    FSDataInputStream input = ifs.open(dataPath);
    Scanner scanner = new Scanner(input, "UTF-8");
    DataConverter converter = new DataConverter(dataset);
    
    int id = 0;
    while (scanner.hasNextLine()) {
      if (id % 1000 == 0) {
        log.info("progress : {}", id);
      }
      
      String line = scanner.nextLine();
      if (line.isEmpty()) {
        continue; // skip empty lines
      }
      
      // write the tuple in files[tuple.label]
      Instance instance = converter.convert(line);
      int label = (int) dataset.getLabel(instance);
      files[currents[label]].writeBytes(line);
      files[currents[label]].writeChar('\n');
      
      // update currents
      currents[label]++;
      if (currents[label] == numPartitions) {
        currents[label] = 0;
      }
    }
    
    // close all the files.
    scanner.close();
    for (FSDataOutputStream file : files) {
      Closeables.close(file, false);
    }
    
    // merge all output files
    FileUtil.copyMerge(pfs, partsPath, fs, outputPath, true, conf, null);
    /*
     * FSDataOutputStream joined = fs.create(new Path(outputPath, "uniform.data")); for (int p = 0; p <
     * numPartitions; p++) {log.info("Joining part : {}", p); FSDataInputStream partStream =
     * fs.open(partPaths[p]);
     * 
     * IOUtils.copyBytes(partStream, joined, conf, false);
     * 
     * partStream.close(); }
     * 
     * joined.close();
     * 
     * fs.delete(partsPath, true);
     */
  }
  
}
