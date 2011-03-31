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
import java.util.Iterator;
import java.util.NoSuchElementException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;

/**
 * <p>{@link Iterator} over a {@link SequenceFile}'s values only.</p>
 */
public final class SequenceFileValueIterator<V extends Writable> implements Iterator<V>, Closeable {

  private final SequenceFile.Reader reader;
  private final Configuration conf;
  private final Class<V> valueClass;
  private final Writable key;
  private V value;
  private boolean available;
  private final boolean reuseKeyValueInstances;

  /**
   * @throws IOException if path can't be read, or its key or value class can't be instantiated
   */
  public SequenceFileValueIterator(Path path, boolean reuseKeyValueInstances, Configuration conf) throws IOException {
    value = null;
    FileSystem fs = path.getFileSystem(conf);
    path = path.makeQualified(fs);
    reader = new SequenceFile.Reader(fs, path, conf);
    this.conf = conf;
    Class<? extends Writable> keyClass = (Class<? extends Writable>) reader.getKeyClass();
    key = ReflectionUtils.newInstance(keyClass, conf);
    valueClass = (Class<V>) reader.getValueClass();
    available = false;
    this.reuseKeyValueInstances = reuseKeyValueInstances;
  }

  public Class<V> getValueClass() {
    return valueClass;
  }

  @Override
  public void close() throws IOException {
    available = false;
    value = null;
    reader.close();
  }

  @Override
  public boolean hasNext() {
    if (!available) {
      if (!reuseKeyValueInstances || value == null) {
        value = ReflectionUtils.newInstance(valueClass, conf);
      }
      try {
        available = reader.next(key, value);
        if (!available) {
          close();
        }
        return available;
      } catch (IOException ioe) {
        try {
          close();
        } catch (IOException ioe2) {
          throw new IllegalStateException(ioe2);
        }
        throw new IllegalStateException(ioe);
      }
    }
    return available;
  }

  /**
   * @throws IllegalStateException if path can't be read, or its key or value class can't be instantiated
   */
  @Override
  public V next() {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }
    available = false;
    return value;
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }

}