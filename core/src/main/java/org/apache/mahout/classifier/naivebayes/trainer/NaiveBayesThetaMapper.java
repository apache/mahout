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

package org.apache.mahout.classifier.naivebayes.trainer;

import java.io.IOException;
import java.net.URI;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.naivebayes.BayesConstants;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

public class NaiveBayesThetaMapper extends Mapper<IntWritable, VectorWritable, Text, VectorWritable> {
  
  private final OpenObjectIntHashMap<String> labelMap = new OpenObjectIntHashMap<String>();
  private Vector featureSum;
  private Vector labelSum;
  private Vector perLabelThetaNormalizer;
  private double alphaI = 1.0;
  private double vocabCount;
  
  @Override
  protected void map(IntWritable key, VectorWritable value, Context context) throws IOException, InterruptedException {
    Vector vector = value.get();
    int label = key.get();
    double weight = Math.log((vector.zSum() + alphaI) / (labelSum.get(label) + vocabCount));
    perLabelThetaNormalizer.set(label, perLabelThetaNormalizer.get(label) + weight);
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    URI[] localFiles = DistributedCache.getCacheFiles(conf);
    Preconditions.checkArgument(localFiles != null && localFiles.length >= 2,
        "missing paths from the DistributedCache");

    alphaI = conf.getFloat(NaiveBayesTrainer.ALPHA_I, 1.0f);
    Path weightFile = new Path(localFiles[0].getPath());

    for (Pair<Text,VectorWritable> record
         : new SequenceFileIterable<Text,VectorWritable>(weightFile, true, conf)) {
      Text key = record.getFirst();
      VectorWritable value = record.getSecond();
      if (key.toString().equals(BayesConstants.FEATURE_SUM)) {
        featureSum = value.get();
      } else  if (key.toString().equals(BayesConstants.LABEL_SUM)) {
        labelSum = value.get();
      }
    }
    perLabelThetaNormalizer = labelSum.like();
    vocabCount = featureSum.getNumNondefaultElements();

    Path labelMapFile = new Path(localFiles[1].getPath());

    // key is word value is id
    for (Pair<Writable,IntWritable> record 
         : new SequenceFileIterable<Writable,IntWritable>(labelMapFile, true, conf)) {
      labelMap.put(record.getFirst().toString(), record.getSecond().get());
    }
  }
  
  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    context.write(new Text(BayesConstants.LABEL_THETA_NORMALIZER), new VectorWritable(perLabelThetaNormalizer));
    super.cleanup(context);
  }
}
