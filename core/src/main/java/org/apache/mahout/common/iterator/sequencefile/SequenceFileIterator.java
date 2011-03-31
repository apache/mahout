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
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.common.Pair;

/**
 * <p>{@link Iterator} over a {@link SequenceFile}'s keys and values, as a {@link Pair}
 * containing key and value.</p>
 */
public final class SequenceFileIterator<K extends Writable,V extends Writable>
  implements Iterator<Pair<K,V>>, Closeable {

  private final SequenceFile.Reader reader;
  private final Configuration conf;
  private final Class<K> keyClass;
  private final Class<V> valueClass;
  private final boolean noValue;
  private K key;
  private V value;
  private boolean available;
  private final boolean reuseKeyValueInstances;

  /**
   * @throws IOException if path can't be read, or its key or value class can't be instantiated
   */
  public SequenceFileIterator(Path path, boolean reuseKeyValueInstances, Configuration conf) throws IOException {
    key = null;
    value = null;
    FileSystem fs = path.getFileSystem(conf);
    path = path.makeQualified(fs);
    reader = new SequenceFile.Reader(fs, path, conf);
    this.conf = conf;
    keyClass = (Class<K>) reader.getKeyClass();
    valueClass = (Class<V>) reader.getValueClass();
    available = false;
    noValue = NullWritable.class.equals(valueClass);
    this.reuseKeyValueInstances = reuseKeyValueInstances;
  }

  public Class<K> getKeyClass() {
    return keyClass;
  }

  public Class<V> getValueClass() {
    return valueClass;
  }

  @Override
  public void close() throws IOException {
    available = false;
    key = null;
    value = null;
    reader.close();
  }

  @Override
  public boolean hasNext() {
    if (!available) {
      if (!reuseKeyValueInstances || value == null) {
        key = ReflectionUtils.newInstance(keyClass, conf);
        if (!noValue) {
          value = ReflectionUtils.newInstance(valueClass, conf);
        }
      }
      try {
        if (noValue) {
          available = reader.next(key);
        } else {
          available = reader.next(key, value);
        }
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
  public Pair<K,V> next() {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }
    available = false;
    return new Pair<K,V>(key, value);
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }

}