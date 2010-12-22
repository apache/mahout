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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class HadoopUtil {
  
  private static final Logger log = LoggerFactory.getLogger(HadoopUtil.class);
  
  private HadoopUtil() { }
  
  public static void overwriteOutput(Path output) throws IOException {
    FileSystem fs = FileSystem.get(output.toUri(), new Configuration());
    //boolean wasFile = fs.isFile(output);
    if (fs.exists(output)) {
      log.info("Deleting {}", output);
      fs.delete(output, true);
    }
    //if (!wasFile) {
    //  log.info("Creating dir {}", output);
    //  fs.mkdirs(output);
    //}
  }
  
  public static void deletePaths(Iterable<Path> paths, FileSystem fs) throws IOException {
    for (Path path : paths) {
      if (fs.exists(path)) {
        log.info("Deleting {}", path);
        fs.delete(path, true);
      }
    }
  }

}
