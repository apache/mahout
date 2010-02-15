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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class HadoopUtil {
  
  private static final Logger log = LoggerFactory.getLogger(HadoopUtil.class);
  
  private HadoopUtil() { }
  
  public static void overwriteOutput(String output) throws IOException {
    Configuration conf = new JobConf(KMeansDriver.class);
    Path outPath = new Path(output);
    FileSystem fs = FileSystem.get(outPath.toUri(), conf);
    if (fs.exists(outPath)) {
      log.warn("Deleting {}", outPath);
      fs.delete(outPath, true);
    }
    log.warn("Creating dir {}", outPath);
    fs.mkdirs(outPath);
  }
  
  public static void deletePath(String output, FileSystem fs) throws IOException {
    Path outPath = new Path(output);
    if (fs.exists(outPath)) {
      log.warn("Deleting {}", outPath);
      fs.delete(outPath, true);
    }
  }
  
  public static void deletePaths(Iterable<Path> paths, FileSystem fs) throws IOException {
    for (Path path : paths) {
      if (fs.exists(path)) {
        log.warn("Deleting {}", path);
        fs.delete(path, true);
      }
    }
  }
  
  public static void rename(Path from, Path to, FileSystem fs) throws IOException {
    log.warn("Renaming " + from.toUri() + " to " + to.toUri());
    fs.rename(from, to);
  }
}
