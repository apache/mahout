package org.apache.mahout.classifier.bayes.common;
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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericsUtil;
import org.apache.mahout.classifier.bayes.io.SequenceFileModelReader;

import java.util.HashMap;
import java.util.Map;

/**
 * The Driver which drives the Tf-Idf Generation
 *
 **/
public class BayesTfIdfDriver {
  /**
   * Takes in two arguments:
   * <ol>
   * <li>The input {@link org.apache.hadoop.fs.Path} where the input documents live</li>
   * <li>The output {@link org.apache.hadoop.fs.Path} where to write the interim files as a {@link org.apache.hadoop.io.SequenceFile}</li>
   * </ol>
   * @param args The args
   */
  public static void main(String[] args) {
    String input = args[0];
    String output = args[1];

    runJob(input, output);
  }

  /**
   * Run the job
   *
   * @param input            the input pathname String
   * @param output           the output pathname String
   */
  public static void runJob(String input, String output) {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(BayesTfIdfDriver.class);
    

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(FloatWritable.class);
    
    SequenceFileInputFormat.addInputPath(conf, new Path(output + "/trainer-termDocCount"));
    SequenceFileInputFormat.addInputPath(conf, new Path(output + "/trainer-wordFreq"));
    SequenceFileInputFormat.addInputPath(conf, new Path(output + "/trainer-featureCount"));
    Path outPath = new Path(output + "/trainer-tfIdf");
    SequenceFileOutputFormat.setOutputPath(conf, outPath);
    conf.setNumMapTasks(100);
    
    conf.setMapperClass(BayesTfIdfMapper.class);
    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setCombinerClass(BayesTfIdfReducer.class);
    conf.setReducerClass(BayesTfIdfReducer.class);    
    conf.setOutputFormat(BayesTfIdfOutputFormat.class);
    
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization"); // Dont ever forget this. People should keep track of how hadoop conf parameters and make or break a piece of code
     try {
      FileSystem dfs = FileSystem.get(conf);
      if (dfs.exists(outPath))
        dfs.delete(outPath, true);
      
      SequenceFileModelReader reader = new SequenceFileModelReader();
      
      Path interimFile = new Path(output+"/trainer-docCount/part-*");      
      
      HashMap<String,Float> labelDocumentCounts= reader.readLabelDocumentCounts(dfs, interimFile, conf);

      DefaultStringifier<HashMap<String,Float>> mapStringifier = new DefaultStringifier<HashMap<String,Float>>(conf,GenericsUtil.getClass(labelDocumentCounts));
      
      String labelDocumentCountString = mapStringifier.toString(labelDocumentCounts);
      System.out.println("Counts of documents in Each Label");
      Map<String,Float> c = mapStringifier.fromString(labelDocumentCountString);
      System.out.println(c);
      
      conf.set("cnaivebayes.labelDocumentCounts", labelDocumentCountString);
      
      client.setConf(conf);    
    
      JobClient.runJob(conf);      
      
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    
  }
}
