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

package org.apache.mahout.clustering.syntheticcontrol.canopy;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.mahout.matrix.Vector;

import java.io.IOException;

public class InputDriver {
  private InputDriver() {
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException {
    String input = args[0];
    String output = args[1];
    String vectorClassName = args[2];
    Class<? extends Vector> vectorClass = (Class<? extends Vector>) Class.forName(vectorClassName);
    runJob(input, output, vectorClass);
  }

  public static void runJob(String input, String output, Class<? extends Vector> vectorClass) throws IOException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(InputDriver.class);

    conf.setOutputKeyClass(Text.class);
    conf.setOutputValueClass(vectorClass);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    FileInputFormat.setInputPaths(conf, new Path(input));
    FileOutputFormat.setOutputPath(conf, new Path(output));

    conf.setMapperClass(InputMapper.class);

    conf.setReducerClass(Reducer.class);
    conf.setNumReduceTasks(0);

    client.setConf(conf);
    JobClient.runJob(conf);
  }

}
