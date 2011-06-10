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
import java.util.Collection;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class MeanShiftCanopyReducer extends Reducer<Text,MeanShiftCanopy,Text,MeanShiftCanopy> {
  
  private final Collection<MeanShiftCanopy> canopies = Lists.newArrayList();
  private MeanShiftCanopyClusterer clusterer;
  private boolean allConverged = true;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    clusterer = new MeanShiftCanopyClusterer(context.getConfiguration());
  }

  @Override
  protected void reduce(Text key, Iterable<MeanShiftCanopy> values, Context context)
    throws IOException, InterruptedException {
    for (MeanShiftCanopy value : values) {
      clusterer.mergeCanopy(value.shallowCopy(), canopies);
    }
    
    for (MeanShiftCanopy canopy : canopies) {
      boolean converged = clusterer.shiftToMean(canopy);
      if (converged) {
        context.getCounter("Clustering", "Converged Clusters").increment(1);
      }
      allConverged = converged && allConverged;
      context.write(new Text(canopy.getIdentifier()), canopy);
    }
    
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    if (allConverged) {
      Path path = new Path(conf.get(MeanShiftCanopyConfigKeys.CONTROL_PATH_KEY));
      FileSystem.get(conf).createNewFile(path);
    }
    super.cleanup(context);
  }
}
