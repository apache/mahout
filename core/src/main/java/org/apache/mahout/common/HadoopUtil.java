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
import java.net.URI;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
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

  /**
   * Count all the records in a directory using a {@link org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator}
   * @param path The {@link org.apache.hadoop.fs.Path} to count
   * @param pt The {@link org.apache.mahout.common.iterator.sequencefile.PathType}
   * @param filter Apply the {@link org.apache.hadoop.fs.PathFilter}.  May be null
   * @param conf The Hadoop {@link org.apache.hadoop.conf.Configuration}
   * @return The number of records
   * @throws IOException if there was an IO error
   */
  public static long countRecords(Path path, PathType pt, PathFilter filter, Configuration conf) throws IOException {
    long count = 0;
    Iterator<?> iterator = new SequenceFileDirValueIterator<Writable>(path, pt, filter, null, true, conf);
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

  public static FileStatus[] getFileStatus(Path path, PathType pathType, PathFilter filter, Comparator<FileStatus> ordering, Configuration conf) throws IOException {
    FileStatus[] statuses;
    FileSystem fs = path.getFileSystem(conf);
    if (filter == null) {
      statuses = pathType == PathType.GLOB ? fs.globStatus(path) : fs.listStatus(path);
    } else {
      statuses = pathType == PathType.GLOB ? fs.globStatus(path, filter) : fs.listStatus(path, filter);
    }
    if (ordering != null) {
      Arrays.sort(statuses, ordering);
    }
    return statuses;
  }

  public static void cacheFiles(Path fileToCache, Configuration conf) {
    DistributedCache.setCacheFiles(new URI[]{fileToCache.toUri()}, conf);
  }

  public static Path cachedFile(Configuration conf) throws IOException {
    return new Path(DistributedCache.getCacheFiles(conf)[0].getPath());
  }

  public static void setSerializations(Configuration conf) {
    conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,"
        + "org.apache.hadoop.io.serializer.WritableSerialization");
  }
}
