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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;

public class MeanShiftCanopyCreatorMapper extends Mapper<WritableComparable<?>, VectorWritable, Text, MeanShiftCanopy> {

  private static final Pattern UNDERSCORE_PATTERN = Pattern.compile("_");

  private static int nextCanopyId = -1;

  /* (non-Javadoc)
   * @see org.apache.hadoop.mapreduce.Mapper#map(java.lang.Object, java.lang.Object, org.apache.hadoop.mapreduce.Mapper.Context)
   */
  @Override
  protected void map(WritableComparable<?> key, VectorWritable point, Context context) throws IOException, InterruptedException {
    MeanShiftCanopy canopy = new MeanShiftCanopy(point.get(), nextCanopyId++);
    context.write(new Text(key.toString()), canopy);
  }

  /* (non-Javadoc)
   * @see org.apache.hadoop.mapreduce.Mapper#setup(org.apache.hadoop.mapreduce.Mapper.Context)
   */
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    if (nextCanopyId == -1) {
      String taskId = context.getConfiguration().get("mapred.task.id");
      String[] parts = UNDERSCORE_PATTERN.split(taskId);
      if (parts.length != 6 || !parts[0].equals("attempt") || (!"m".equals(parts[3]) && !"r".equals(parts[3]))) {
        throw new IllegalArgumentException("TaskAttemptId string : " + taskId + " is not properly formed");
      }
      nextCanopyId = ((1 << 31) / 50000) * (Integer.parseInt(parts[4]));
      //each mapper has 42,949 ids to give.
    }
  }
}
