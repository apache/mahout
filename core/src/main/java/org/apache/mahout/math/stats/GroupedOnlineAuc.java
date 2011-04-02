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

package org.apache.mahout.math.stats;

import com.google.common.collect.Maps;
import org.apache.mahout.classifier.sgd.PolymorphicWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Map;

/**
 * Implements a variant on AUC where the result returned is an average of several AUC measurements
 * made on sub-groups of the overall data.  Controlling for the grouping factor allows the effects
 * of the grouping factor on the model to be ignored.  This is useful, for instance, when using a
 * classifier as a click prediction engine.  In that case you want AUC to refer only to the ranking
 * of items for a particular user, not to the discrimination of users from each other.  Grouping by
 * user (or user cluster) helps avoid optimizing for the wrong quality.
 */
public class GroupedOnlineAuc implements OnlineAuc {
  private final Map<String, OnlineAuc> map = Maps.newHashMap();
  private GlobalOnlineAuc.ReplacementPolicy policy;
  private int windowSize;

  @Override
  public double addSample(int category, String groupKey, double score) {
    if (groupKey == null) {
      addSample(category, score);
    }
    
    OnlineAuc group = map.get(groupKey);
    if (group == null) {
      group = new GlobalOnlineAuc();
      if (policy != null) {
        group.setPolicy(policy);
      }
      if (windowSize > 0) {
        group.setWindowSize(windowSize);
      }
      map.put(groupKey, group);
    }
    return group.addSample(category, score);
  }

  @Override
  public double addSample(int category, double score) {
    throw new UnsupportedOperationException("Can't add to " + this.getClass() + " without group key");
  }

  @Override
  public double auc() {
    double sum = 0;
    for (OnlineAuc auc : map.values()) {
      sum += auc.auc();
    }
    return sum / map.size();
  }

  @Override
  public void setPolicy(GlobalOnlineAuc.ReplacementPolicy policy) {
    this.policy = policy;
    for (OnlineAuc auc : map.values()) {
      auc.setPolicy(policy);
    }
  }

  @Override
  public void setWindowSize(int windowSize) {
    this.windowSize = windowSize;
    for (OnlineAuc auc : map.values()) {
      auc.setWindowSize(windowSize);
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(map.size());
    for (Map.Entry<String,OnlineAuc> entry : map.entrySet()) {
      out.writeUTF(entry.getKey());
      PolymorphicWritable.write(out, entry.getValue());
    }
    out.writeInt(policy.ordinal());
    out.writeInt(windowSize);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int n = in.readInt();
    map.clear();
    for (int i = 0; i < n; i++) {
      String key = in.readUTF();
      map.put(key, PolymorphicWritable.read(in, OnlineAuc.class));
    }
    policy = GlobalOnlineAuc.ReplacementPolicy.values()[in.readInt()];
    windowSize = in.readInt();
  }
}
