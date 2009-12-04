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

package org.apache.mahout.cf.taste.hadoop.pseudo;

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
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.InputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputFormat;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.hadoop.util.StringUtils;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * <p>This job runs a "pseudo-distributed" recommendation process on Hadoop.
 * It merely runs many {@link Recommender} instances on Hadoop, where each instance
 * is a normal non-distributed implementation.</p>
 *
 * <p>This class configures and runs a {@link RecommenderMapper} using Hadoop.</p>
 *
 * <p>Command line arguments are:</p>
 *
 * <ol>
 *  <li>Fully-qualified class name of {@link Recommender} to use to make
 *   recommendations. Note that it must have a constructor which takes a
 *   {@link org.apache.mahout.cf.taste.model.DataModel} argument.</li>
 *  <li>Number of recommendations to compute per user</li>
 *  <li>Location of a text file containing user IDs
 *   for which recommendations should be computed, one per line</li>
 *  <li>Location of a data model file containing preference data,
 *   suitable for use with {@link org.apache.mahout.cf.taste.impl.model.file.FileDataModel}</li>
 *  <li>Output path where reducer output should go</li>
 *  <li>JAR file containing implementation code</li>
 * </ol>
 *
 * <p>Example arguments:</p>
 *
 * {@code
 * --recommenderClassName org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender
 * --userRec 10 --userIdFile path/to/users.txt --dataModelFile path/to/data.csv
 * --output path/to/reducerOutputDir --jarFile recommender.jar
 * }
 *
 * <p>For example, to get started trying this out, set up Hadoop in a
 * pseudo-distributed manner: http://hadoop.apache.org/common/docs/current/quickstart.html
 * You can stop at the point where it instructs you to copy files into HDFS. Instead, proceed as follow.</p>
 *
 * {@code
 * hadoop fs -mkdir input
 * hadoop fs -mkdir output
 * }
 *
 * <p>Assume your preference data file is <code>input.csv</code>. You will also need to create a file
 * containing all user IDs to write recommendations for, as something like <code>users.txt</code>.
 * Place this input on HDFS like so:</p>
 *
 * {@code
 * hadoop fs -put input.csv input/input.csv
 * hadoop fs -put users.txt input/users.txt
 * }
 *
 * <p>Build Mahout code with <code>mvn package</code> in the core/ directory. Locate
 * <code>target/mahout-core-X.Y-SNAPSHOT.job</code>. This is a JAR file; copy it out
 * to a convenient location and name it <code>recommender.jar</code>.</p>
 *
 * <p>Now add your own custom recommender code and dependencies. Your IDE produced compiled .class
 * files somewhere and they need to be packaged up as well:</p>
 *
 * {@code
 * jar uf recommender.jar -C (your classes directory) .
 * }
 *
 * <p>And launch:</p>
 *
 * {@code
 * hadoop jar recommender.jar org.apache.mahout.cf.taste.hadoop.RecommenderJob \
 *   --recommenderClassName your.project.Recommender \
 *   --userRec 10 --userIdFile input/users.txt --dataModelFile input/input.csv \
 *   --output output --jarFile recommender.jar
 * }
 */
public final class RecommenderJob {

  private static final Logger log = LoggerFactory.getLogger(RecommenderJob.class);

  private RecommenderJob() {
  }

  public static void main(String[] args) throws Exception {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option recommendClassOpt = obuilder.withLongName("recommenderClassName").withRequired(true)
      .withShortName("r").withArgument(abuilder.withName("recommenderClassName").withMinimum(1)
      .withMaximum(1).create()).withDescription("Name of recommender class to use.").create();

    Option userRecommendOpt = obuilder.withLongName("userRec").withRequired(true)
      .withShortName("n").withArgument(abuilder.withName("userRec").withMinimum(1)
      .withMaximum(1).create()).withDescription("Desired number of recommendations per user.").create();

    Option userIDFileOpt = obuilder.withLongName("userIdFile").withRequired(true)
      .withShortName("f").withArgument(abuilder.withName("userIdFile").withMinimum(1)
      .withMaximum(1).create()).withDescription("File containing user ids.").create();

    Option dataModelFileOpt = obuilder.withLongName("dataModelFile").withRequired(true)
      .withShortName("m").withArgument(abuilder.withName("dataModelFile").withMinimum(1)
      .withMaximum(1).create()).withDescription("File containing data model.").create();

    Option jarFileOpt = obuilder.withLongName("jarFile").withRequired(true)
      .withShortName("m").withArgument(abuilder.withName("jarFile").withMinimum(1)
      .withMaximum(1).create()).withDescription("Implementation jar.").create();

    Option outputOpt = DefaultOptionCreator.outputOption(obuilder, abuilder).create();
    Option helpOpt = DefaultOptionCreator.helpOption(obuilder);

    Group group = gbuilder.withName("Options").withOption(recommendClassOpt).withOption(userRecommendOpt)
      .withOption(userIDFileOpt).withOption(dataModelFileOpt).withOption(outputOpt)
      .withOption(jarFileOpt).withOption(helpOpt).create();


    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }

      String recommendClassName = cmdLine.getValue(recommendClassOpt).toString();
      int recommendationsPerUser = Integer.parseInt(cmdLine.getValue(userRecommendOpt).toString());
      String userIDFile = cmdLine.getValue(userIDFileOpt).toString();
      String dataModelFile = cmdLine.getValue(dataModelFileOpt).toString();
      String jarFile = cmdLine.getValue(jarFileOpt).toString();
      String outputPath = cmdLine.getValue(outputOpt).toString();
      JobConf jobConf =
          buildJobConf(recommendClassName, recommendationsPerUser, userIDFile, dataModelFile, jarFile, outputPath);
      JobClient.runJob(jobConf);
    } catch (OptionException e) {
      log.error(e.getMessage());
      CommandLineUtil.printHelp(group);
    }
  }

  public static JobConf buildJobConf(String recommendClassName,
                                     int recommendationsPerUser,
                                     String userIDFile,
                                     String dataModelFile,
                                     String jarFile,
                                     String outputPath) throws IOException {

    JobConf jobConf = new JobConf();
    FileSystem fs = FileSystem.get(jobConf);

    Path userIDFilePath = new Path(userIDFile).makeQualified(fs);
    Path outputPathPath = new Path(outputPath).makeQualified(fs);

    if (fs.exists(outputPathPath)) {
      fs.delete(outputPathPath, true);
    }

    jobConf.set("mapred.jar", jarFile);
    jobConf.setJar(jarFile);

    jobConf.set(RecommenderMapper.RECOMMENDER_CLASS_NAME, recommendClassName);
    jobConf.set(RecommenderMapper.RECOMMENDATIONS_PER_USER, String.valueOf(recommendationsPerUser));
    jobConf.set(RecommenderMapper.DATA_MODEL_FILE, dataModelFile);

    jobConf.setClass("mapred.input.format.class", TextInputFormat.class, InputFormat.class);
    jobConf.set("mapred.input.dir", StringUtils.escapeString(userIDFilePath.toString()));

    jobConf.setClass("mapred.mapper.class", RecommenderMapper.class, Mapper.class);
    jobConf.setClass("mapred.mapoutput.key.class", LongWritable.class, Object.class);
    jobConf.setClass("mapred.mapoutput.value.class", RecommendedItemsWritable.class, Object.class);

    jobConf.setClass("mapred.reducer.class", IdentityReducer.class, Reducer.class);
    jobConf.setClass("mapred.output.key.class", LongWritable.class, Object.class);
    jobConf.setClass("mapred.output.value.class", RecommendedItemsWritable.class, Object.class);

    jobConf.setClass("mapred.output.format.class", TextOutputFormat.class, OutputFormat.class);
    jobConf.set("mapred.output.dir", StringUtils.escapeString(outputPathPath.toString()));

    return jobConf;
  }

}
