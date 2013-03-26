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

package org.apache.mahout.math.hadoop.similarity;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;

final class SeedVectorUtil {

  private static final Logger log = LoggerFactory.getLogger(SeedVectorUtil.class);

  private SeedVectorUtil() {
  }

  public static List<NamedVector> loadSeedVectors(Configuration conf) {

    String seedPathStr = conf.get(VectorDistanceSimilarityJob.SEEDS_PATH_KEY);
    if (seedPathStr == null || seedPathStr.isEmpty()) {
      return Collections.emptyList();
    }

    List<NamedVector> seedVectors = Lists.newArrayList();
    long item = 0;
    for (Writable value
        : new SequenceFileDirValueIterable<Writable>(new Path(seedPathStr),
                                                    PathType.LIST,
                                                    PathFilters.partFilter(),
                                                    conf)) {
      Class<? extends Writable> valueClass = value.getClass();
      if (valueClass.equals(Kluster.class)) {
        // get the cluster info
        Kluster cluster = (Kluster) value;
        Vector vector = cluster.getCenter();
        if (vector instanceof NamedVector) {
          seedVectors.add((NamedVector) vector);
        } else {
          seedVectors.add(new NamedVector(vector, cluster.getIdentifier()));
        }
      } else if (valueClass.equals(Canopy.class)) {
        // get the cluster info
        Canopy canopy = (Canopy) value;
        Vector vector = canopy.getCenter();
        if (vector instanceof NamedVector) {
          seedVectors.add((NamedVector) vector);
        } else {
          seedVectors.add(new NamedVector(vector, canopy.getIdentifier()));
        }
      } else if (valueClass.equals(Vector.class)) {
        Vector vector = (Vector) value;
        if (vector instanceof NamedVector) {
          seedVectors.add((NamedVector) vector);
        } else {
          seedVectors.add(new NamedVector(vector, seedPathStr + '.' + item++));
        }
      } else if (valueClass.equals(VectorWritable.class) || valueClass.isInstance(VectorWritable.class)) {
        VectorWritable vw = (VectorWritable) value;
        Vector vector = vw.get();
        if (vector instanceof NamedVector) {
          seedVectors.add((NamedVector) vector);
        } else {
          seedVectors.add(new NamedVector(vector, seedPathStr + '.' + item++));
        }
      } else {
        throw new IllegalStateException("Bad value class: " + valueClass);
      }
    }
    if (seedVectors.isEmpty()) {
      throw new IllegalStateException("No seeds found. Check your path: " + seedPathStr);
    }
    log.info("Seed Vectors size: {}", seedVectors.size());
    return seedVectors;
  }

}
