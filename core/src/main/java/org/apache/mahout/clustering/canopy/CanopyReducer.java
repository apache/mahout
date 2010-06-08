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

package org.apache.mahout.clustering.canopy;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class CanopyReducer extends Reducer<Text, VectorWritable, Text, Canopy> {

  /* (non-Javadoc)
   * @see org.apache.hadoop.mapreduce.Reducer#reduce(java.lang.Object, java.lang.Iterable, org.apache.hadoop.mapreduce.Reducer.Context)
   */
  @Override
  protected void reduce(Text arg0, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {
    Iterator<VectorWritable> it = values.iterator();
    while (it.hasNext()) {
      Vector point = it.next().get();
      canopyClusterer.addPointToCanopies(point, canopies, context);
    }
    for (Canopy canopy : canopies) {
      context.write(new Text(canopy.getIdentifier()), canopy);
    }
  }

  /* (non-Javadoc)
   * @see org.apache.hadoop.mapreduce.Reducer#setup(org.apache.hadoop.mapreduce.Reducer.Context)
   */
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    canopyClusterer = new CanopyClusterer(context.getConfiguration());
  }

  private final List<Canopy> canopies = new ArrayList<Canopy>();

  private CanopyClusterer canopyClusterer;

}
