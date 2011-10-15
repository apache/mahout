/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.mahout.classifier.bayes.mapreduce.common;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.mahout.common.StringTuple;

import com.google.common.base.Preconditions;

import java.io.Serializable;

public class FeatureLabelComparator extends WritableComparator implements Serializable {

  public FeatureLabelComparator() {
    super(StringTuple.class, true);
  }
  
  @Override
  public int compare(WritableComparable a, WritableComparable b) {
    StringTuple ta = (StringTuple) a;
    StringTuple tb = (StringTuple) b;
    Preconditions.checkArgument(ta.length() >= 2 && ta.length() <= 3 && tb.length() >= 2 && tb.length() <= 3,
                                "StringTuple length out of bounds");
    // token
    String tmpa = ta.length() == 2 ? ta.stringAt(1) : ta.stringAt(2);
    String tmpb = tb.length() == 2 ? tb.stringAt(1) : tb.stringAt(2);
    int cmp = tmpa.compareTo(tmpb);
    if (cmp != 0) {
      return cmp;
    }
    
    // type, FEATURE_TF first, then FEATURE_COUNT, then DF or anything else.
    cmp = ta.stringAt(0).compareTo(tb.stringAt(0));
    if (cmp != 0) {
      if (ta.stringAt(0).equals(BayesConstants.FEATURE_TF)) {
        return -1;
      }
      if (tb.stringAt(0).equals(BayesConstants.FEATURE_TF)) {
        return 1;
      }
      if (ta.stringAt(0).equals(BayesConstants.FEATURE_COUNT)) {
        return -1;
      }
      if (tb.stringAt(0).equals(BayesConstants.FEATURE_COUNT)) {
        return 1;
      }
      return cmp;
    }

    // label or empty.
    tmpa = ta.length() == 2 ? "" : ta.stringAt(1);
    tmpb = tb.length() == 2 ? "" : tb.stringAt(1);
    
    return tmpa.compareTo(tmpb);
  }
  
}
