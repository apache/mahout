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

package org.apache.mahout.common;

import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;

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
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.Tool;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>Superclass of many Mahout Hadoop "jobs". A job drives configuration and launch of one or
 * more maps and reduces in order to accomplish some task.</p>
 *
 * <p>Command line arguments available to all subclasses are:</p>
 *
 * <ul>
 *  <li>--tempDir (path): Specifies a directory where the job may place temp files
 *   (default "temp")</li>
 *  <li>--help: Show help message</li>
 * </ul>
 *
 * <p>In addition, note some key command line parameters that are parsed by Hadoop, which jobs
 * may need to set:</p>
 *
 * <ul>
 *  <li>-Dmapred.job.name=(name): Sets the Hadoop task names. It will be suffixed by
 *    the mapper and reducer class names</li>
 *  <li>-Dmapred.output.compress={true,false}: Compress final output (default true)</li>
 *  <li>-Dmapred.input.dir=(path): input file, or directory containing input files (required)</li>
 *  <li>-Dmapred.output.dir=(path): path to write output files (required)</li>
 * </ul>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public abstract class AbstractJob extends Configured implements Tool {
  
  private static final Logger log = LoggerFactory.getLogger(AbstractJob.class);

  protected static Option buildOption(String name, String shortName, String description) {
    return buildOption(name, shortName, description, true, null);
  }
  
  protected static Option buildOption(String name, String shortName, String description, String defaultValue) {
    return buildOption(name, shortName, description, false, defaultValue);
  }

  protected static Option buildOption(String name, String shortName, String description,
                                      boolean required) {
    return buildOption(name, shortName, description, required, null);
  }
  
  protected static Option buildOption(String name,
                                    String shortName,
                                    String description,
                                    boolean required,
                                    String defaultValue) {
    ArgumentBuilder argBuilder = new ArgumentBuilder().withName(name).withMinimum(1).withMaximum(1);
    if (defaultValue != null) {
      argBuilder = argBuilder.withDefault(defaultValue);
    }
    Argument arg = argBuilder.create();
    DefaultOptionBuilder optBuilder = new DefaultOptionBuilder().withLongName(name).withRequired(required)
        .withArgument(arg).withDescription(description);
    if (shortName != null) {
      optBuilder = optBuilder.withShortName(shortName);
    }
    return optBuilder.create();
  }
  
  protected static Map<String,String> parseArguments(String[] args, Option... extraOpts) {
    
    Option tempDirOpt = buildOption("tempDir", null, "Intermediate output directory", "temp");
    Option helpOpt = DefaultOptionCreator.helpOption();
    Option startPhase = buildOption("startPhase", null, "First phase to run", "0");
    Option endPhase = buildOption("endPhase", null, "Last phase to run", String.valueOf(Integer.MAX_VALUE));

    GroupBuilder gBuilder = new GroupBuilder().withName("Options")
        .withOption(tempDirOpt)
        .withOption(helpOpt)
        .withOption(startPhase).withOption(endPhase);
    
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
    
    Map<String,String> result = new TreeMap<String,String>();
    maybePut(result, cmdLine, tempDirOpt, helpOpt, startPhase, endPhase);
    maybePut(result, cmdLine, extraOpts);

    log.info("Command line arguments: {}", result);
    return result;
  }
  
  protected static void maybePut(Map<String,String> args, CommandLine cmdLine, Option... opt) {
    for (Option o : opt) {
      Object value = cmdLine.getValue(o);
      if (value != null) {
        args.put(o.getPreferredName(), value.toString());
      }
    }
  }

  protected static boolean shouldRunNextPhase(Map<String,String> args, AtomicInteger currentPhase) {
    int phase = currentPhase.getAndIncrement();
    String startPhase = args.get("--startPhase");
    String endPhase = args.get("--endPhase");
    boolean phaseSkipped = (startPhase != null && phase < Integer.parseInt(startPhase)) 
        || (endPhase != null && phase > Integer.parseInt(endPhase));
    if (phaseSkipped) {
      log.info("Skipping phase {}", phase);
    }
    return !phaseSkipped;
  }
  
  protected Job prepareJob(Path inputPath,
                           Path outputPath,
                           Class<? extends InputFormat> inputFormat,
                           Class<? extends Mapper> mapper,
                           Class<? extends Writable> mapperKey,
                           Class<? extends Writable> mapperValue,
                           Class<? extends Reducer> reducer,
                           Class<? extends Writable> reducerKey,
                           Class<? extends Writable> reducerValue,
                           Class<? extends OutputFormat> outputFormat) throws IOException {
    
    Job job = new Job(new Configuration(getConf()));
    Configuration jobConf = job.getConfiguration();

    if (reducer.equals(Reducer.class)) {
      if (mapper.equals(Mapper.class)) {
        throw new IllegalStateException("Can't figure out the user class jar file from mapper/reducer");
      }
      job.setJarByClass(mapper);
    } else {
      job.setJarByClass(reducer);
    }

    job.setInputFormatClass(inputFormat);
    jobConf.set("mapred.input.dir", inputPath.toString());

    job.setMapperClass(mapper);
    job.setMapOutputKeyClass(mapperKey);
    job.setMapOutputValueClass(mapperValue);

    jobConf.setBoolean("mapred.compress.map.output", true);

    job.setReducerClass(reducer);
    job.setOutputKeyClass(reducerKey);
    job.setOutputValueClass(reducerValue);

    String customJobName = job.getJobName();
    if (customJobName == null || customJobName.trim().length() == 0) {
      customJobName = getClass().getSimpleName();
    }
    customJobName += '-' + mapper.getSimpleName();
    customJobName += '-' + reducer.getSimpleName();
    job.setJobName(customJobName);

    job.setOutputFormatClass(outputFormat);
    jobConf.set("mapred.output.dir", outputPath.toString());
    
    return job;
  }
  
}
