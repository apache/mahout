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

package org.apache.mahout.cf.taste.hadoop.als;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import org.apache.mahout.math.map.OpenIntObjectHashMap;

import java.io.IOException;

/** Solving mapper that can be safely executed using multiple threads */
public class SolveExplicitFeedbackMapper
    extends SharingMapper<IntWritable,VectorWritable,IntWritable,VectorWritable,OpenIntObjectHashMap<Vector>> {

  private double lambda;
  private int numFeatures;
  private final VectorWritable uiOrmj = new VectorWritable();

  @Override
  OpenIntObjectHashMap<Vector> createSharedInstance(Context ctx) throws IOException {
    Configuration conf = ctx.getConfiguration();
    int numEntities = Integer.parseInt(conf.get(ParallelALSFactorizationJob.NUM_ENTITIES));
    return ALS.readMatrixByRowsFromDistributedCache(numEntities, conf);
  }

  @Override
  protected void setup(Mapper.Context ctx) throws IOException, InterruptedException {
    lambda = Double.parseDouble(ctx.getConfiguration().get(ParallelALSFactorizationJob.LAMBDA));
    numFeatures = ctx.getConfiguration().getInt(ParallelALSFactorizationJob.NUM_FEATURES, -1);
    Preconditions.checkArgument(numFeatures > 0, "numFeatures must be greater then 0!");
  }

  @Override
  protected void map(IntWritable userOrItemID, VectorWritable ratingsWritable, Context ctx)
    throws IOException, InterruptedException {
    OpenIntObjectHashMap<Vector> uOrM = getSharedInstance();
    uiOrmj.set(ALS.solveExplicit(ratingsWritable, uOrM, lambda, numFeatures));
    ctx.write(userOrItemID, uiOrmj);
  }

}
