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
import java.io.InputStream;
import java.util.Arrays;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class HadoopUtil {
  
  private static final Logger log = LoggerFactory.getLogger(HadoopUtil.class);
  
  private HadoopUtil() { }

  public static void delete(Configuration conf, Iterable<Path> paths) throws IOException {
    if (conf == null) {
      conf = new Configuration();
    }
    for (Path path : paths) {
      FileSystem fs = path.getFileSystem(conf);
      if (fs.exists(path)) {
        log.info("Deleting {}", path);
        fs.delete(path, true);
      }
    }
  }
  
  public static void delete(Configuration conf, Path... paths) throws IOException {
    delete(conf, Arrays.asList(paths));
  }

  public static long countRecords(Path path, Configuration conf) throws IOException {
    long count = 0;
    Iterator<?> iterator = new SequenceFileValueIterator<Writable>(path, true, conf);
    while (iterator.hasNext()) {
      iterator.next();
      count++;
    }
    return count;
  }

  public static InputStream openStream(Path path, Configuration conf) throws IOException {
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    return fs.open(path.makeQualified(fs));
  }

}
