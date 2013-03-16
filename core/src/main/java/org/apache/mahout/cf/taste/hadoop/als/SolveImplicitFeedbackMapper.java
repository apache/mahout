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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.als.ImplicitFeedbackAlternatingLeastSquaresSolver;
import org.apache.mahout.math.map.OpenIntObjectHashMap;

import java.io.IOException;

class SolveImplicitFeedbackMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

  private ImplicitFeedbackAlternatingLeastSquaresSolver solver;

  private final VectorWritable uiOrmj = new VectorWritable();

  @Override
  protected void setup(Context ctx) throws IOException, InterruptedException {
    double lambda = Double.parseDouble(ctx.getConfiguration().get(ParallelALSFactorizationJob.LAMBDA));
    double alpha = Double.parseDouble(ctx.getConfiguration().get(ParallelALSFactorizationJob.ALPHA));
    int numFeatures = ctx.getConfiguration().getInt(ParallelALSFactorizationJob.NUM_FEATURES, -1);

    Path YPath = new Path(ctx.getConfiguration().get(ParallelALSFactorizationJob.FEATURE_MATRIX));
    OpenIntObjectHashMap<Vector> Y = ALS.readMatrixByRows(YPath, ctx.getConfiguration());

    solver = new ImplicitFeedbackAlternatingLeastSquaresSolver(numFeatures, lambda, alpha, Y);

    Preconditions.checkArgument(numFeatures > 0, "numFeatures was not set correctly!");
  }

  @Override
  protected void map(IntWritable userOrItemID, VectorWritable ratingsWritable, Context ctx)
      throws IOException, InterruptedException {
    uiOrmj.set(solver.solve(ratingsWritable.get()));
    ctx.write(userOrItemID, uiOrmj);
  }
}
