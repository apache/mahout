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

package org.apache.mahout.clustering.cdbw;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.math.VectorWritable;

public class CDbwReducer extends MapReduceBase implements Reducer<IntWritable, WeightedVectorWritable, IntWritable, VectorWritable> {

  private Map<Integer, List<VectorWritable>> referencePoints;

  private OutputCollector<IntWritable, VectorWritable> output;

  @Override
  public void reduce(IntWritable key, Iterator<WeightedVectorWritable> values, OutputCollector<IntWritable, VectorWritable> output,
      Reporter reporter) throws IOException {
    this.output = output;
    // find the most distant point
    WeightedVectorWritable mdp = null;
    while (values.hasNext()) {
      WeightedVectorWritable dpw = values.next();
      if (mdp == null || mdp.getWeight() < dpw.getWeight()) {
        mdp = new WeightedVectorWritable(dpw.getWeight(), dpw.getVector());
      }
    }
    output.collect(new IntWritable(key.get()), mdp.getVector());
  }

  public void configure(Map<Integer, List<VectorWritable>> referencePoints) {
    this.referencePoints = referencePoints;
  }

  /* (non-Javadoc)
   * @see org.apache.hadoop.mapred.MapReduceBase#close()
   */
  @Override
  public void close() throws IOException {
    for (Integer clusterId : referencePoints.keySet()) {
      for (VectorWritable vw : referencePoints.get(clusterId)) {
        output.collect(new IntWritable(clusterId), vw);
      }
    }
    super.close();
  }

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    try {
      referencePoints = CDbwMapper.getRepresentativePoints(job);
    } catch (NumberFormatException e) {
      throw new IllegalStateException(e);
    } catch (SecurityException e) {
      throw new IllegalStateException(e);
    } catch (IllegalArgumentException e) {
      throw new IllegalStateException(e);
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(e);
    } catch (InvocationTargetException e) {
      throw new IllegalStateException(e);
    }
  }

}
