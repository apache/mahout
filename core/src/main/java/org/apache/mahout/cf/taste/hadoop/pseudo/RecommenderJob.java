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

import java.io.IOException;
import java.util.Map;

import org.apache.commons.cli2.Option;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.recommender.Recommender;

/**
 * <p>
 * This job runs a "pseudo-distributed" recommendation process on Hadoop. It merely runs many
 * {@link Recommender} instances on Hadoop, where each instance is a normal non-distributed implementation.
 * </p>
 * 
 * <p>This class configures and runs a {@link RecommenderReducer} using Hadoop.</p>
 * 
 * <p>Command line arguments specific to this class are:</p>
 * 
 * <ol>
 * <li>-Dmapred.input.dir=(path): Location of a data model file containing preference data, suitable for use with
 * {@link org.apache.mahout.cf.taste.impl.model.file.FileDataModel}</li>
 * <li>-Dmapred.output.dir=(path): output path where recommender output should go</li>
 * <li>--recommenderClassName (string): Fully-qualified class name of {@link Recommender} to use to make recommendations.
 * Note that it must have a constructor which takes a {@link org.apache.mahout.cf.taste.model.DataModel}
 * argument.</li>
 * <li>--numRecommendations (integer): Number of recommendations to compute per user</li>
 * <li>--usersFile (path): file containing user IDs to recommend for (optional)</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 * 
 * <p>
 * For example, to get started trying this out, set up Hadoop in a pseudo-distributed manner:
 * http://hadoop.apache.org/common/docs/current/quickstart.html You can stop at the point where it instructs
 * you to copy files into HDFS.
 * </p>
 * 
 * <p>
 * Assume your preference data file is <code>input.csv</code>. You will also need to create a file containing
 * all user IDs to write recommendations for, as something like <code>users.txt</code>. Place this input on
 * HDFS like so:
 * </p>
 * 
 * {@code hadoop fs -put input.csv input/input.csv; hadoop fs -put users.txt input/users.txt * }
 * 
 * <p>
 * Build Mahout code with <code>mvn package</code> in the core/ directory. Locate
 * <code>target/mahout-core-X.Y-SNAPSHOT.job</code>. This is a JAR file; copy it out to a convenient location
 * and name it <code>recommender.jar</code>.
 * </p>
 * 
 * <p>
 * Now add your own custom recommender code and dependencies. Your IDE produced compiled .class files
 * somewhere and they need to be packaged up as well:
 * </p>
 * 
 * {@code jar uf recommender.jar -C (your classes directory) . * }
 * 
 * <p>
 * And launch:
 * </p>
 * 
 * {@code hadoop jar recommender.jar \
 *   org.apache.mahout.cf.taste.hadoop.pseudo.RecommenderJob \
 *   -Dmapred.input.dir=input/users.csv \
 *   -Dmapred.output.dir=output \
 *   --recommenderClassName your.project.Recommender \
 *   --numRecommendations 10  *
 * }
 */
public final class RecommenderJob extends AbstractJob {
  
  @Override
  public int run(String[] args) throws IOException {
    
    Option recommendClassOpt = AbstractJob.buildOption("recommenderClassName", "r",
      "Name of recommender class to instantiate");
    Option numReccomendationsOpt = AbstractJob.buildOption("numRecommendations", "n",
      "Number of recommendations per user", "10");
    Option usersFileOpt = AbstractJob.buildOption("usersFile", "u", "Number of recommendations per user",
      null);
    
    Map<String,String> parsedArgs = AbstractJob.parseArguments(args, recommendClassOpt,
      numReccomendationsOpt, usersFileOpt);
    if (parsedArgs == null) {
      return -1;
    }

    Configuration originalConf = getConf();
    String inputFile = originalConf.get("mapred.input.dir");
    String outputPath = originalConf.get("mapred.output.dir");
    String usersFile = parsedArgs.get("--usersFile");
    if (usersFile == null) {
      usersFile = inputFile;
    }
    
    String recommendClassName = parsedArgs.get("--recommenderClassName");
    int recommendationsPerUser = Integer.parseInt(parsedArgs.get("--numRecommendations"));
    
    JobConf jobConf = prepareJobConf(usersFile, outputPath, TextInputFormat.class,
      UserIDsMapper.class, LongWritable.class, NullWritable.class, RecommenderReducer.class,
      LongWritable.class, RecommendedItemsWritable.class, TextOutputFormat.class);
    
    jobConf.set(RecommenderReducer.RECOMMENDER_CLASS_NAME, recommendClassName);
    jobConf.setInt(RecommenderReducer.RECOMMENDATIONS_PER_USER, recommendationsPerUser);
    jobConf.set(RecommenderReducer.DATA_MODEL_FILE, inputFile);
    jobConf.set(RecommenderReducer.USERS_FILE, usersFile);
    jobConf.setClass("mapred.output.compression.codec", GzipCodec.class, CompressionCodec.class);
    
    JobClient.runJob(jobConf);
    return 0;
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RecommenderJob(), args);
  }
  
}
