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

package org.apache.mahout.ga.watchmaker.cd;

import com.google.common.collect.Lists;

import java.util.Collections;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Represents one line of a Dataset. Contains only real attributs.
 */
public class DataLine {

  private static final Pattern COMMA = Pattern.compile(",");

  private final double[] attributes;
  
  public DataLine() {
    int nba = DataSet.getDataSet().getNbAttributes();
    attributes = new double[nba];
  }
  
  public DataLine(CharSequence dl) {
    this();
    set(dl);
  }
  
  public int getLabel() {
    int labelPos = DataSet.getDataSet().getLabelIndex();
    return (int) attributes[labelPos];
  }
  
  public double getAttribute(int index) {
    return attributes[index];
  }
  
  public void set(CharSequence dlstr) {
    DataSet dataset = DataSet.getDataSet();
    
    // extract tokens
    List<String> tokens = Lists.newArrayList();
    Collections.addAll(tokens, COMMA.split(dlstr));

    // remove any ignored attribute
    List<Integer> ignored = dataset.getIgnoredAttributes();
    for (int index = ignored.size() - 1; index >= 0; index--) {
      tokens.remove(index);
    }
    
    // load attributes
    for (int index = 0; index < dataset.getNbAttributes(); index++) {
      if (dataset.isNumerical(index)) {
        attributes[index] = Double.parseDouble(tokens.get(index));
      } else {
        attributes[index] = dataset.valueIndex(index, tokens.get(index));
      }
    }
  }
}
