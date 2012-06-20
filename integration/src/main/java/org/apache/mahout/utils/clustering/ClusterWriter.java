/*
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

package org.apache.mahout.utils.clustering;

import java.io.Closeable;
import java.io.IOException;

import org.apache.mahout.clustering.iterator.ClusterWritable;

/**
 * Writes out clusters
 */
public interface ClusterWriter extends Closeable {

  /**
   * Write all values in the Iterable to the output
   *
   * @param iterable The {@link Iterable} to loop over
   * @return the number of docs written
   * @throws java.io.IOException if there was a problem writing
   */
  long write(Iterable<ClusterWritable> iterable) throws IOException;

  /**
   * Write out a Cluster
   */
  void write(ClusterWritable clusterWritable) throws IOException;

  /**
   * Write the first {@code maxDocs} to the output.
   *
   * @param iterable The {@link Iterable} to loop over
   * @param maxDocs  the maximum number of docs to write
   * @return The number of docs written
   * @throws IOException if there was a problem writing
   */
  long write(Iterable<ClusterWritable> iterable, long maxDocs) throws IOException;
}
