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

package org.apache.mahout.clustering.conversion;

import java.io.IOException;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class converts text files containing space-delimited floating point numbers into
 * Mahout sequence files of VectorWritable suitable for input to the clustering jobs in
 * particular, and any Mahout job requiring this input in general.
 *
 */
public final class InputDriver {

  private static final Logger log = LoggerFactory.getLogger(InputDriver.class);
  
  private InputDriver() {
  }
  
  public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputOpt = DefaultOptionCreator.inputOption().withRequired(false).create();
    Option outputOpt = DefaultOptionCreator.outputOption().withRequired(false).create();
    Option vectorOpt = obuilder.withLongName("vector").withRequired(false).withArgument(
      abuilder.withName("v").withMinimum(1).withMaximum(1).create()).withDescription(
      "The vector implementation to use.").withShortName("v").create();
    
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(
      vectorOpt).withOption(helpOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      Path input = new Path(cmdLine.getValue(inputOpt, "testdata").toString());
      Path output = new Path(cmdLine.getValue(outputOpt, "output").toString());
      String vectorClassName = cmdLine.getValue(vectorOpt,
         "org.apache.mahout.math.RandomAccessSparseVector").toString();
      runJob(input, output, vectorClassName);
    } catch (OptionException e) {
      log.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }
  }
  
  public static void runJob(Path input, Path output, String vectorClassName)
    throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set("vector.implementation.class.name", vectorClassName);
    Job job = new Job(conf, "Input Driver running over input: " + input);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(InputMapper.class);   
    job.setNumReduceTasks(0);
    job.setJarByClass(InputDriver.class);
    
    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, output);
    
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }
  
}
