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

package org.apache.mahout.classifier.bayes;

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
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.analysis.WikipediaAnalyzer;
import org.apache.mahout.utils.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Set;

/**
 * Create and run the Wikipedia Dataset Creator.
 */
public class WikipediaDatasetCreatorDriver {
  private transient static Logger log = LoggerFactory.getLogger(WikipediaDatasetCreatorDriver.class);

  private WikipediaDatasetCreatorDriver() {
  }

  /**
   * Takes in two arguments:
   * <ol>
   * <li>The input {@link org.apache.hadoop.fs.Path} where the input documents live</li>
   * <li>The output {@link org.apache.hadoop.fs.Path} where to write the
   * {@link org.apache.mahout.classifier.bayes.BayesModel} as a {@link org.apache.hadoop.io.SequenceFile}</li>
   * </ol>
   *
   * @param args The args
   */
  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option dirInputPathOpt = obuilder.withLongName("input").withRequired(true).withArgument(
            abuilder.withName("input").withMinimum(1).withMaximum(1).create()).
            withDescription("The input directory path").withShortName("i").create();

    Option dirOutputPathOpt = obuilder.withLongName("output").withRequired(true).withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
            withDescription("The output directory Path").withShortName("o").create();

    Option categoriesOpt = obuilder.withLongName("categories").withRequired(true).withArgument(
            abuilder.withName("categories").withMinimum(1).withMaximum(1).create()).
            withDescription("Location of the categories file.  One entry per line.  Will be used to make a string match in Wikipedia Category field").withShortName("c").create();

    Option exactMatchOpt = obuilder.withLongName("exactMatch").
            withDescription("If set, then the category name must exactly match the entry in the categories file. Default is false").withShortName("e").create();
    Option analyzerOpt = obuilder.withLongName("analyzer").withRequired(false).withArgument(
            abuilder.withName("analyzer").withMinimum(1).withMaximum(1).create()).
            withDescription("The analyzer to use, must have a no argument constructor").withShortName("a").create();
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(categoriesOpt).withOption(dirInputPathOpt).withOption(dirOutputPathOpt)
            .withOption(exactMatchOpt).withOption(analyzerOpt)
            .withOption(helpOpt).create();

    Parser parser = new Parser();
    parser.setGroup(group);
    CommandLine cmdLine = null;
    try {
      cmdLine = parser.parse(args);
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
        analyzerClass = (Class<? extends Analyzer>) Class.forName(className);
        //try instantiating it, b/c there isn't any point in setting it if
        //you can't instantiate it
        analyzerClass.newInstance();
      }
      runJob(inputPath, outputPath, catFile, cmdLine.hasOption(exactMatchOpt), analyzerClass);
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    } catch (ClassNotFoundException e) {
      log.error("Exception: Analyzer class not found", e);
    } catch (IllegalAccessException e) {
      log.error("Exception: Couldn't instantiate the class", e);
    } catch (InstantiationException e) {
      log.error("Exception: Couldn't instantiate the class", e);
    }
  }

  /**
   * Run the job
   *
   * @param input          the input pathname String
   * @param output         the output pathname String
   * @param catFile        the file containing the Wikipedia categories
   * @param exactMatchOnly if true, then the Wikipedia category must match exactly instead of simply containing the category string
   */
  public static void runJob(String input, String output, String catFile,
                            boolean exactMatchOnly, Class<? extends Analyzer> analyzerClass) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(WikipediaDatasetCreatorDriver.class);
    if (log.isInfoEnabled()) {
      log.info("Input: " + input + " Out: " + output + " Categories: " + catFile);
    }
    conf.set("key.value.separator.in.input.line", " ");
    conf.set("xmlinput.start", "<text xml:space=\"preserve\">");
    conf.set("xmlinput.end", "</text>");
    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(Text.class);
    conf.setBoolean("exact.match.only", exactMatchOnly);
    conf.set("analyzer.class", analyzerClass.getName());
    FileInputFormat.setInputPaths(conf, new Path(input));
    Path outPath = new Path(output);
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setMapperClass(WikipediaDatasetCreatorMapper.class);
    conf.setNumMapTasks(100);
    conf.setInputFormat(XmlInputFormat.class);
    //conf.setCombinerClass(WikipediaDatasetCreatorReducer.class);
    conf.setReducerClass(WikipediaDatasetCreatorReducer.class);
    conf.setOutputFormat(WikipediaDatasetCreatorOutputFormat.class);
    conf.set("io.serializations",
            "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
    // Dont ever forget this. People should keep track of how hadoop conf parameters and make or break a piece of code

    FileSystem dfs = FileSystem.get(outPath.toUri(), conf);
    if (dfs.exists(outPath)) {
      dfs.delete(outPath, true);
    }

    Set<String> categories = new HashSet<String>();
    BufferedReader reader = new BufferedReader(new InputStreamReader(
            new FileInputStream(catFile), "UTF-8"));
    String line;
    while ((line = reader.readLine()) != null) {
      categories.add(line.trim().toLowerCase());
    }
    reader.close();

    DefaultStringifier<Set<String>> setStringifier = new DefaultStringifier<Set<String>>(conf, GenericsUtil.getClass(categories));

    String categoriesStr = setStringifier.toString(categories);

    conf.set("wikipedia.categories", categoriesStr);

    client.setConf(conf);
    JobClient.runJob(conf);
  }
}
