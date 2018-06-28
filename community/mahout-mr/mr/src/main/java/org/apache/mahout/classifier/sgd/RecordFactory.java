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

package org.apache.mahout.classifier.sgd;

import org.apache.mahout.math.Vector;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A record factor understands how to convert a line of data into fields and then into a vector.
 */
public interface RecordFactory {
  void defineTargetCategories(List<String> values);

  RecordFactory maxTargetValue(int max);

  boolean usesFirstLineAsSchema();

  int processLine(String line, Vector featureVector);

  Iterable<String> getPredictors();

  Map<String, Set<Integer>> getTraceDictionary();

  RecordFactory includeBiasTerm(boolean useBias);

  List<String> getTargetCategories();

  void firstLine(String line);
}
