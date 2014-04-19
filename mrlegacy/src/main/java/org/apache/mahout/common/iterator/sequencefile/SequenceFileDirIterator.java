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

import java.io.Closeable;
import java.io.IOException;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.ForwardingIterator;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.Pair;

/**
 * Like {@link SequenceFileIterator}, but iterates not just over one sequence file, but many. The input path
 * may be specified as a directory of files to read, or as a glob pattern. The set of files may be optionally
 * restricted with a {@link PathFilter}.
 */
public final class SequenceFileDirIterator<K extends Writable,V extends Writable>
    extends ForwardingIterator<Pair<K,V>> implements Closeable {

  private static final FileStatus[] NO_STATUSES = new FileStatus[0];

  private Iterator<Pair<K,V>> delegate;
  private final List<SequenceFileIterator<K,V>> iterators;

  /**
   * Multifile sequence file iterator where files are specified explicitly by
   * path parameters.
   */
  public SequenceFileDirIterator(Path[] path,
                                 boolean reuseKeyValueInstances,
                                 Configuration conf) throws IOException {

    iterators = Lists.newArrayList();
    // we assume all files should exist, otherwise we will bail out.
    FileSystem fs = FileSystem.get(path[0].toUri(), conf);
    FileStatus[] statuses = new FileStatus[path.length];
    for (int i = 0; i < statuses.length; i++) {
      statuses[i] = fs.getFileStatus(path[i]);
    }
    init(statuses, reuseKeyValueInstances, conf);
  }

  /**
   * Constructor that uses either {@link FileSystem#listStatus(Path)} or
   * {@link FileSystem#globStatus(Path)} to obtain list of files to iterate over
   * (depending on pathType parameter).
   */
  public SequenceFileDirIterator(Path path,
                                 PathType pathType,
                                 PathFilter filter,
                                 Comparator<FileStatus> ordering,
                                 boolean reuseKeyValueInstances,
                                 Configuration conf) throws IOException {

    FileStatus[] statuses = HadoopUtil.getFileStatus(path, pathType, filter, ordering, conf);
    iterators = Lists.newArrayList();
    init(statuses, reuseKeyValueInstances, conf);
  }

  private void init(FileStatus[] statuses,
                    final boolean reuseKeyValueInstances,
                    final Configuration conf) {

    /*
     * prevent NPEs. Unfortunately, Hadoop would return null for list if nothing
     * was qualified. In this case, which is a corner case, we should assume an
     * empty iterator, not an NPE.
     */
    if (statuses == null) {
      statuses = NO_STATUSES;
    }

    Iterator<FileStatus> fileStatusIterator = Iterators.forArray(statuses);

    Iterator<Iterator<Pair<K, V>>> fsIterators =
      Iterators.transform(fileStatusIterator,
        new Function<FileStatus, Iterator<Pair<K, V>>>() {
          @Override
          public Iterator<Pair<K, V>> apply(FileStatus from) {
            try {
              SequenceFileIterator<K, V> iterator = new SequenceFileIterator<K, V>(from.getPath(),
                  reuseKeyValueInstances, conf);
              iterators.add(iterator);
              return iterator;
            } catch (IOException ioe) {
              throw new IllegalStateException(from.getPath().toString(), ioe);
            }
          }
        });

    Collections.reverse(iterators); // close later in reverse order

    delegate = Iterators.concat(fsIterators);
  }

  @Override
  protected Iterator<Pair<K,V>> delegate() {
    return delegate;
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(iterators);
    iterators.clear();
  }

}
