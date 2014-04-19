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

package org.apache.mahout.fpm.pfpgrowth.convertors;

import java.io.IOException;

import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.OutputCollector;

/**
 * Collects the {@link Writable} key and {@link Writable} value, and writes them into a {@link SequenceFile}
 * 
 * @param <K>
 * @param <V>
 */
public class SequenceFileOutputCollector<K extends Writable,V extends Writable> implements
    OutputCollector<K,V> {
  private final SequenceFile.Writer writer;
  
  public SequenceFileOutputCollector(SequenceFile.Writer writer) {
    this.writer = writer;
  }
  
  @Override
  public final void collect(K key, V value) throws IOException {
    writer.append(key, value);
  }
  
}
