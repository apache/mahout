/*
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

package org.apache.mahout.text.wikipedia;

import java.io.File;
import java.io.IOException;
import java.util.Locale;
import java.util.Set;

import com.google.common.collect.Sets;
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
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.Stringifier;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Create and run the Wikipedia Dataset Creator.
 */
public final class WikipediaDatasetCreatorDriver {
  private static final Logger log = LoggerFactory.getLogger(WikipediaDatasetCreatorDriver.class);
  
  private WikipediaDatasetCreatorDriver() { }
  
  /**
   * Takes in two arguments:
   * <ol>
   * <li>The input {@link org.apache.hadoop.fs.Path} where the input documents live</li>
   * <li>The output {@link org.apache.hadoop.fs.Path} where to write the classifier as a
   * {@link org.apache.hadoop.io.SequenceFile}</li>
   * </ol>
   */
  public static void main(String[] args) throws IOException, InterruptedException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option dirInputPathOpt = DefaultOptionCreator.inputOption().create();
    
    Option dirOutputPathOpt = DefaultOptionCreator.outputOption().create();
    
    Option categoriesOpt = obuilder.withLongName("categories").withRequired(true).withArgument(
      abuilder.withName("categories").withMinimum(1).withMaximum(1).create()).withDescription(
      "Location of the categories file.  One entry per line. "
          + "Will be used to make a string match in Wikipedia Category field").withShortName("c").create();
    
    Option exactMatchOpt = obuilder.withLongName("exactMatch").withDescription(
      "If set, then the category name must exactly match the "
          + "entry in the categories file. Default is false").withShortName("e").create();
    Option analyzerOpt = obuilder.withLongName("analyzer").withRequired(false).withArgument(
      abuilder.withName("analyzer").withMinimum(1).withMaximum(1).create()).withDescription(
      "The analyzer to use, must have a no argument constructor").withShortName("a").create();
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    Group group = gbuilder.withName("Options").withOption(categoriesOpt).withOption(dirInputPathOpt)
        .withOption(dirOutputPathOpt).withOption(exactMatchOpt).withOption(analyzerOpt).withOption(helpOpt)
        .create();
    
    Parser parser = new Parser();
    parser.setGroup(group);
    try {
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      String inputPath = (String) cmdLine.getValue(dirInputPathOpt);
      String outputPath = (String) cmdLine.getValue(dirOutputPathOpt);
      String catFile = (String) cmdLine.getValue(categoriesOpt);
      Class<? extends Analyzer> analyzerClass = WikipediaAnalyzer.class;
      if (cmdLine.hasOption(analyzerOpt)) {
        String className = cmdLine.getValue(analyzerOpt).toString();
        analyzerClass = Class.forName(className).asSubclass(Analyzer.class);
        // try instantiating it, b/c there isn't any point in setting it if
        // you can't instantiate it
        ClassUtils.instantiateAs(analyzerClass, Analyzer.class);
      }
      runJob(inputPath, outputPath, catFile, cmdLine.hasOption(exactMatchOpt),
        analyzerClass);
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    } catch (ClassNotFoundException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }
  
  /**
   * Run the job
   * 
   * @param input
   *          the input pathname String
   * @param output
   *          the output pathname String
   * @param catFile
   *          the file containing the Wikipedia categories
   * @param exactMatchOnly
   *          if true, then the Wikipedia category must match exactly instead of simply containing the
   *          category string
   */
  public static void runJob(String input,
                            String output,
                            String catFile,
                            boolean exactMatchOnly,
                            Class<? extends Analyzer> analyzerClass)
    throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set("key.value.separator.in.input.line", " ");
    conf.set("xmlinput.start", "<page>");
    conf.set("xmlinput.end", "</page>");
    conf.setBoolean("exact.match.only", exactMatchOnly);
    conf.set("analyzer.class", analyzerClass.getName());
    conf.set("io.serializations",
             "org.apache.hadoop.io.serializer.JavaSerialization,"
             + "org.apache.hadoop.io.serializer.WritableSerialization");
    // Dont ever forget this. People should keep track of how hadoop conf
    // parameters can make or break a piece of code
    
    Set<String> categories = Sets.newHashSet();
    for (String line : new FileLineIterable(new File(catFile))) {
      categories.add(line.trim().toLowerCase(Locale.ENGLISH));
    }
    
    Stringifier<Set<String>> setStringifier =
        new DefaultStringifier<Set<String>>(conf, GenericsUtil.getClass(categories));
    
    String categoriesStr = setStringifier.toString(categories);
    
    conf.set("wikipedia.categories", categoriesStr);
    
    Job job = new Job(conf);
    log.info("Input: {} Out: {} Categories: {}", input, output, catFile);
    job.setJarByClass(WikipediaDatasetCreatorDriver.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);
    job.setMapperClass(WikipediaDatasetCreatorMapper.class);
    //TODO: job.setNumMapTasks(100);
    job.setInputFormatClass(XmlInputFormat.class);
    job.setReducerClass(WikipediaDatasetCreatorReducer.class);
    job.setOutputFormatClass(TextOutputFormat.class);
    
    FileInputFormat.setInputPaths(job, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(job, outPath);
    HadoopUtil.delete(conf, outPath);
    
    boolean succeeded = job.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }
}
