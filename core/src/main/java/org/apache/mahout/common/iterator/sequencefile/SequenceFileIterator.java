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

import com.google.common.collect.AbstractIterator;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.common.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>{@link java.util.Iterator} over a {@link SequenceFile}'s keys and values, as a {@link Pair}
 * containing key and value.</p>
 */
public final class SequenceFileIterator<K extends Writable,V extends Writable>
  extends AbstractIterator<Pair<K,V>> implements Closeable {

  private final SequenceFile.Reader reader;
  private final Configuration conf;
  private final Class<K> keyClass;
  private final Class<V> valueClass;
  private final boolean noValue;
  private K key;
  private V value;
  private final boolean reuseKeyValueInstances;

  private static final Logger log = LoggerFactory.getLogger(SequenceFileIterator.class);

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
    key = null;
    value = null;
    Closeables.close(reader, true);

    endOfData();
  }

  @Override
  protected Pair<K,V> computeNext() {
    if (!reuseKeyValueInstances || value == null) {
      key = ReflectionUtils.newInstance(keyClass, conf);
      if (!noValue) {
        value = ReflectionUtils.newInstance(valueClass, conf);
      }
    }
    try {
      boolean available;
      if (noValue) {
        available = reader.next(key);
      } else {
        available = reader.next(key, value);
      }
      if (!available) {
        close();
        return null;
      }
      return new Pair<K,V>(key, value);
    } catch (IOException ioe) {
      try {
        close();
      } catch (IOException e) {
        log.error(e.getMessage(), e);
      }
      throw new IllegalStateException(ioe);
    }
  }

}
