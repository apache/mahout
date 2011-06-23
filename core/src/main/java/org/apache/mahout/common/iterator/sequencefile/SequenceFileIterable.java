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
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.Pair;

/**
 * <p>{@link Iterable} counterpart to {@link SequenceFileIterator}.</p>
 */
public final class SequenceFileIterable<K extends Writable,V extends Writable> implements Iterable<Pair<K,V>> {

  private final Path path;
  private final boolean reuseKeyValueInstances;
  private final Configuration conf;

  /**
   * Like {@link #SequenceFileIterable(Path, boolean, Configuration)} but key and value instances are not reused
   * by default.
   *
   * @param path file to iterate over
   */
  public SequenceFileIterable(Path path, Configuration conf) {
    this(path, false, conf);
  }

  /**
   * @param path file to iterate over
   * @param reuseKeyValueInstances if true, reuses instances of the key and value object instead of creating a new
   *  one for each read from the file
   */
  public SequenceFileIterable(Path path, boolean reuseKeyValueInstances, Configuration conf) {
    this.path = path;
    this.reuseKeyValueInstances = reuseKeyValueInstances;
    this.conf = conf;
  }

  @Override
  public Iterator<Pair<K, V>> iterator() {
    try {
      return new SequenceFileIterator<K, V>(path, reuseKeyValueInstances, conf);
    } catch (IOException ioe) {
      throw new IllegalStateException(path.toString(), ioe);
    }
  }

}

