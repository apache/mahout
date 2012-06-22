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

package org.apache.mahout.utils.vectors.arff;

import java.text.DateFormat;
import java.util.Map;


/**
 * An interface for representing an ARFFModel.  Implementations can decide on the best approach
 * for storing the model, as some approaches will be fine for smaller files, while larger
 * ones may require a better implementation.
 */
public interface ARFFModel {
  String ARFF_SPARSE = "{"; //indicates the vector is sparse
  String ARFF_SPARSE_END = "}";
  String ARFF_COMMENT = "%";
  String ATTRIBUTE = "@attribute";
  String DATA = "@data";
  String RELATION = "@relation";
  
  
  String getRelation();
  
  void setRelation(String relation);
  
  /**
   * The vector attributes (labels in Mahout speak)
   * @return the map
   */
  Map<String, Integer> getLabelBindings();
  
  Integer getNominalValue(String label, String nominal);
  
  void addNominal(String label, String nominal, int idx);
  
  DateFormat getDateFormat(Integer idx);
  
  void addDateFormat(Integer idx, DateFormat format);
  
  Integer getLabelIndex(String label);
  
  void addLabel(String label, Integer idx);
  
  ARFFType getARFFType(Integer idx);
  
  void addType(Integer idx, ARFFType type);
  
  /**
   * The count of the number of words seen
   * @return the count
   */
  long getWordCount();
  
  double getValue(String data, int idx);
  
  Map<String, Map<String, Integer>> getNominalMap();
  
  int getLabelSize();
  
  Map<String, Long> getWords();
}
