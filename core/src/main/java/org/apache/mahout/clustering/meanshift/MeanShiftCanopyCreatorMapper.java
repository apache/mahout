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

package org.apache.mahout.clustering.meanshift;

import java.io.IOException;
import java.util.regex.Pattern;

import com.google.common.base.Preconditions;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.kmeans.KMeansConfigKeys;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.VectorWritable;

public class MeanShiftCanopyCreatorMapper extends Mapper<WritableComparable<?>, VectorWritable, Text, MeanShiftCanopy> {

  private static final Pattern UNDERSCORE_PATTERN = Pattern.compile("_");

  private static int nextCanopyId = -1;

  private DistanceMeasure measure;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable point, Context context)
    throws IOException, InterruptedException {
    MeanShiftCanopy canopy = MeanShiftCanopy.initialCanopy(point.get(), nextCanopyId++, measure);
    context.write(new Text(key.toString()), canopy);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    String measureClass = context.getConfiguration().get(KMeansConfigKeys.DISTANCE_MEASURE_KEY);
    ClassLoader ccl = Thread.currentThread().getContextClassLoader();
    try {
      measure = ccl.loadClass(measureClass).asSubclass(DistanceMeasure.class).newInstance();
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    }

    if (nextCanopyId == -1) {
      String taskId = context.getConfiguration().get("mapred.task.id");
      String[] parts = UNDERSCORE_PATTERN.split(taskId);
      Preconditions.checkArgument(parts.length == 6
          && "attempt".equals(parts[0])
          && ("m".equals(parts[3]) || "r".equals(parts[3])),
          "TaskAttemptId string: %d is not properly formed", taskId);
      nextCanopyId = ((1 << 31) / 50000) * Integer.parseInt(parts[4]);
      //each mapper has 42,949 ids to give.
    }
  }
}
