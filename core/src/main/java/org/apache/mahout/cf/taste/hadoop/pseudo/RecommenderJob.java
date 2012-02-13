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

package org.apache.mahout.cf.taste.hadoop.pseudo;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.math.VarLongWritable;

/**
 * <p>
 * This job runs a "pseudo-distributed" recommendation process on Hadoop. It merely runs many
 * {@link org.apache.mahout.cf.taste.recommender.Recommender} instances on Hadoop,
 * where each instance is a normal non-distributed implementation.
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
 * <li>--recommenderClassName (string): Fully-qualified class name of
 * {@link org.apache.mahout.cf.taste.recommender.Recommender} to use to make recommendations.
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
 * Assume your preference data file is {@code input.csv}. You will also need to create a file containing
 * all user IDs to write recommendations for, as something like {@code users.txt}. Place this input on
 * HDFS like so:
 * </p>
 *
 * {@code hadoop fs -put input.csv input/input.csv; hadoop fs -put users.txt input/users.txt * }
 *
 * <p>
 * Build Mahout code with {@code mvn package} in the core/ directory. Locate
 * {@code target/mahout-core-X.Y-SNAPSHOT.job}. This is a JAR file; copy it out to a convenient location
 * and name it {@code recommender.jar}.
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
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

    addInputOption();
    addOutputOption();
    addOption("recommenderClassName", "r", "Name of recommender class to instantiate");
    addOption("numRecommendations", "n", "Number of recommendations per user", "10");
    addOption("usersFile", "u", "File of users to recommend for", null);
    
    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path inputFile = getInputPath();
    Path outputPath = getOutputPath();
    Path usersFile = hasOption("usersFile") ? inputFile : new Path(getOption("usersFile"));
    
    String recommendClassName = getOption("recommenderClassName");
    int recommendationsPerUser = Integer.parseInt(getOption("numRecommendations"));
    
    Job job = prepareJob(usersFile,
                         outputPath,
                         TextInputFormat.class,
                         UserIDsMapper.class,
                         VarLongWritable.class,
                         NullWritable.class,
                         RecommenderReducer.class,
                         VarLongWritable.class,
                         RecommendedItemsWritable.class,
                         TextOutputFormat.class);
    FileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
    Configuration jobConf = job.getConfiguration();
    jobConf.set(RecommenderReducer.RECOMMENDER_CLASS_NAME, recommendClassName);
    jobConf.setInt(RecommenderReducer.RECOMMENDATIONS_PER_USER, recommendationsPerUser);
    jobConf.set(RecommenderReducer.DATA_MODEL_FILE, inputFile.toString());

    boolean succeeded = job.waitForCompletion(true);
    return succeeded ? 0 : -1;
  }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RecommenderJob(), args);
  }
  
}
