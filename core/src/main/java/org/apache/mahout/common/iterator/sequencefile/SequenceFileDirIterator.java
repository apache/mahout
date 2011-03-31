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

package org.apache.mahout.common.iterator.sequencefile;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.ArrayIterator;
import org.apache.mahout.common.iterator.DelegatingIterator;
import org.apache.mahout.common.iterator.IteratorsIterator;
import org.apache.mahout.common.iterator.TransformingIterator;

/**
 * Like {@link SequenceFileIterator}, but iterates not just over one sequence file, but many. The input path
 * may be specified as a directory of files to read, or as a glob pattern. The set of files may be optionally
 * restricted with a {@link PathFilter}.
 */
public final class SequenceFileDirIterator<K extends Writable,V extends Writable>
    extends DelegatingIterator<Pair<K,V>> {

  public SequenceFileDirIterator(Path path,
                                 PathType pathType,
                                 PathFilter filter,
                                 Comparator<FileStatus> ordering,
                                 boolean reuseKeyValueInstances,
                                 Configuration conf)
    throws IOException {
    super(SequenceFileDirIterator.<K,V>buildDelegate(path,
                                                     pathType,
                                                     filter,
                                                     ordering,
                                                     reuseKeyValueInstances,
                                                     conf));
  }

  private static <K extends Writable,V extends Writable> Iterator<Pair<K,V>> buildDelegate(
      Path path,
      PathType pathType,
      PathFilter filter,
      Comparator<FileStatus> ordering,
      boolean reuseKeyValueInstances,
      Configuration conf) throws IOException {

    FileSystem fs = path.getFileSystem(conf);
    path = path.makeQualified(fs);
    FileStatus[] statuses;
    if (filter == null) {
      statuses = pathType == PathType.GLOB ? fs.globStatus(path) : fs.listStatus(path);
    } else {
      statuses = pathType == PathType.GLOB ? fs.globStatus(path, filter) : fs.listStatus(path, filter);
    }
    if (ordering != null) {
      Arrays.sort(statuses, ordering);
    }
    Iterator<FileStatus> fileStatusIterator = new ArrayIterator<FileStatus>(statuses);
    return new IteratorsIterator<Pair<K,V>>(
        new FileStatusToSFIterator<K,V>(fileStatusIterator, reuseKeyValueInstances, conf));
  }


  private static class FileStatusToSFIterator<K extends Writable, V extends Writable>
    extends TransformingIterator<FileStatus,Iterator<Pair<K,V>>> {

    private final Configuration conf;
    private final boolean reuseKeyValueInstances;

    private FileStatusToSFIterator(Iterator<FileStatus> fileStatusIterator,
                                   boolean reuseKeyValueInstances,
                                   Configuration conf) {
      super(fileStatusIterator);
      this.reuseKeyValueInstances = reuseKeyValueInstances;
      this.conf = conf;
    }

    @Override
    protected Iterator<Pair<K,V>> transform(FileStatus in) {
      try {
        return new SequenceFileIterator<K,V>(in.getPath(), reuseKeyValueInstances, conf);
      } catch (IOException ioe) {
        throw new IllegalStateException(ioe);
      }
    }
  }

}
