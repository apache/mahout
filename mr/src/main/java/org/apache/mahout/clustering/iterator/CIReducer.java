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

package org.apache.mahout.clustering.iterator;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;

public class CIReducer extends Reducer<IntWritable,ClusterWritable,IntWritable,ClusterWritable> {
  
  private ClusterClassifier classifier;
  private ClusteringPolicy policy;
  
  @Override
  protected void reduce(IntWritable key, Iterable<ClusterWritable> values, Context context) throws IOException,
      InterruptedException {
    Iterator<ClusterWritable> iter = values.iterator();
    Cluster first = iter.next().getValue(); // there must always be at least one
    while (iter.hasNext()) {
      Cluster cluster = iter.next().getValue();
      first.observe(cluster);
    }
    List<Cluster> models = Lists.newArrayList();
    models.add(first);
    classifier = new ClusterClassifier(models, policy);
    classifier.close();
    context.write(key, new ClusterWritable(first));
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String priorClustersPath = conf.get(ClusterIterator.PRIOR_PATH_KEY);
    classifier = new ClusterClassifier();
    classifier.readFromSeqFiles(conf, new Path(priorClustersPath));
    policy = classifier.getPolicy();
    policy.update(classifier);
    super.setup(context);
  }
  
}
