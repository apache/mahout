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

package org.apache.mahout.utils.clustering;

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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.matrix.Vector;
import org.apache.mahout.utils.CommandLineUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public final class ClusterDumper {

  private static final Logger log = LoggerFactory.getLogger(ClusterDumper.class);
  private static final String LINE_SEP = System.getProperty("line.separator");

  private ClusterDumper() {
  }

  public static void main(String[] args) throws IOException, IllegalAccessException, InstantiationException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option seqOpt = obuilder.withLongName("seqFile").withRequired(false).withArgument(
            abuilder.withName("seqFile").withMinimum(1).withMaximum(1).create()).
            withDescription("The Sequence File containing the Clusters").withShortName("s").create();
    Option outputOpt = obuilder.withLongName("output").withRequired(false).withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
            withDescription("The output file.  If not specified, dumps to the console").withShortName("o").create();
    Option substringOpt = obuilder.withLongName("substring").withRequired(false).withArgument(
            abuilder.withName("substring").withMinimum(1).withMaximum(1).create()).
            withDescription("The number of chars of the asFormatString() to print").withShortName("b").create();
    Option pointsOpt = obuilder.withLongName("points").withRequired(false).withArgument(
            abuilder.withName("points").withMinimum(1).withMaximum(1).create()).
            withDescription("The points sequence file mapping input vectors to their cluster.  If specified, then the program will output the points associated with a cluster").withShortName("p").create();
    Option helpOpt = obuilder.withLongName("help").
            withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(seqOpt).withOption(outputOpt).withOption(substringOpt).withOption(pointsOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {

        CommandLineUtil.printHelp(group);
        return;
      }

      if (cmdLine.hasOption(seqOpt)) {
        Path path = new Path(cmdLine.getValue(seqOpt).toString());
        System.out.println("Input Path: " + path);
        JobClient client = new JobClient();
        JobConf conf = new JobConf(Job.class);
        client.setConf(conf);
        FileSystem fs = FileSystem.get(path.toUri(), conf);
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
        Map<String, List<String>> clusterIdToPoints = null;
        if (cmdLine.hasOption(pointsOpt)) {
          //read in the points
          clusterIdToPoints = readPoints(cmdLine.getValue(pointsOpt).toString(), conf);
        } else {
          clusterIdToPoints = Collections.emptyMap();
        }
        Writer writer = null;
        if (cmdLine.hasOption(outputOpt)){
          writer = new FileWriter(cmdLine.getValue(outputOpt).toString());
        } else {
          writer = new OutputStreamWriter(System.out);
        }
        int sub = Integer.MAX_VALUE;
        if (cmdLine.hasOption(substringOpt)) {
          sub = Integer.parseInt(cmdLine.getValue(substringOpt).toString());
        }
        Writable key = (Writable) reader.getKeyClass().newInstance();
        ClusterBase value = (ClusterBase) reader.getValueClass().newInstance();
        while (reader.next(key, value)){
          Vector center = value.getCenter();
          String fmtStr = center.asFormatString();
          writer.append(String.valueOf(value.getId())).append(":").append("name:")
                  .append(center.getName()).append(":").append(fmtStr.substring(0, Math.min(sub, fmtStr.length()))).append(LINE_SEP);
          List<String> points = clusterIdToPoints.get(String.valueOf(value.getId()));
          if (points != null){
            writer.write("\tPoints: ");
            for (Iterator<String> iterator = points.iterator(); iterator.hasNext();) {
              String point = iterator.next();
              writer.append(point);
              if (iterator.hasNext()){
                writer.append(", ");
              }
            }
            writer.write(LINE_SEP);
          }
          writer.flush();
        }
        if (cmdLine.hasOption(outputOpt)){
          writer.flush();
          writer.close();
        }
      }

    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }

  }

  private static Map<String, List<String>> readPoints(String pointsPath, JobConf conf) throws IOException {
    Map<String, List<String>> result = new HashMap<String, List<String>>();
    Path path = new Path(pointsPath);
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    try {
      Text key = (Text) reader.getKeyClass().newInstance();
      Text value = (Text) reader.getValueClass().newInstance();
      while (reader.next(key, value)) {
        //value is the cluster id as an int, key is the name/id of the vector, but that doesn't matter because we only care about printing it
        String clusterId = value.toString();
        List<String> pointList = result.get(clusterId);
        if (pointList == null) {
          pointList = new ArrayList<String>();
          result.put(clusterId, pointList);
        }
        pointList.add(key.toString());

      }
    } catch (InstantiationException e) {
      log.error("Exception", e);
    } catch (IllegalAccessException e) {
      log.error("Exception", e);
    }
    return result;
  }


}