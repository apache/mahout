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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ReflectionUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>{@link java.util.Iterator} over a {@link SequenceFile}'s values only.</p>
 */
public final class SequenceFileValueIterator<V extends Writable> extends AbstractIterator<V> implements Closeable {

  private final SequenceFile.Reader reader;
  private final Configuration conf;
  private final Class<V> valueClass;
  private final Writable key;
  private V value;
  private final boolean reuseKeyValueInstances;

  private static final Logger log = LoggerFactory.getLogger(SequenceFileValueIterator.class);

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
    this.reuseKeyValueInstances = reuseKeyValueInstances;
  }

  public Class<V> getValueClass() {
    return valueClass;
  }

  @Override
  public void close() throws IOException {
    value = null;
    Closeables.close(reader, true);
    endOfData();
  }

  @Override
  protected V computeNext() {
    if (!reuseKeyValueInstances || value == null) {
      value = ReflectionUtils.newInstance(valueClass, conf);
    }
    try {
      boolean available = reader.next(key, value);
      if (!available) {
        close();
        return null;
      }
      return value;
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
