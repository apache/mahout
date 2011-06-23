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
import java.util.Comparator;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.Writable;

/**
 * <p>{@link Iterable} counterpart to {@link SequenceFileDirValueIterator}.</p>
 */
public final class SequenceFileDirValueIterable<V extends Writable> implements Iterable<V> {

  private final Path path;
  private final PathType pathType;
  private final PathFilter filter;
  private final Comparator<FileStatus> ordering;
  private final boolean reuseKeyValueInstances;
  private final Configuration conf;

  public SequenceFileDirValueIterable(Path path, PathType pathType, Configuration conf) {
    this(path, pathType, null, conf);
  }

  public SequenceFileDirValueIterable(Path path, PathType pathType, PathFilter filter, Configuration conf) {
    this(path, pathType, filter, null, false, conf);
  }

  /**
   * @param path file to iterate over
   * @param pathType whether or not to treat path as a directory ({@link PathType#LIST}) or
   *  glob pattern ({@link PathType#GLOB})
   * @param filter if not null, specifies sequence files to be ignored by the iteration
   * @param ordering if not null, specifies the order in which to iterate over matching sequence files
   * @param reuseKeyValueInstances if true, reuses instances of the value object instead of creating a new
   *  one for each read from the file
   */
  public SequenceFileDirValueIterable(Path path,
                                      PathType pathType,
                                      PathFilter filter,
                                      Comparator<FileStatus> ordering,
                                      boolean reuseKeyValueInstances,
                                      Configuration conf) {
    this.path = path;
    this.pathType = pathType;
    this.filter = filter;
    this.ordering = ordering;
    this.reuseKeyValueInstances = reuseKeyValueInstances;
    this.conf = conf;
  }

  @Override
  public Iterator<V> iterator() {
    try {
      return new SequenceFileDirValueIterator<V>(path, pathType, filter, ordering, reuseKeyValueInstances, conf);
    } catch (IOException ioe) {
      throw new IllegalStateException(path.toString(), ioe);
    }
  }

}

