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

package org.apache.mahout.common.parameters;

/**
 * An accessor to a parameters in the job.
 * 
 * This is a composite entity that can it self contain more parameters. Say the parameters describes what
 * DistanceMeasure class to use, once set this parameters would also produce the parameters available in that
 * DistanceMeasure implementation.
 */
public interface Parameter<T> extends Parametered {
  /** @return job configuration setting key prefix, e.g. 'org.apache.mahout.util.WeightedDistanceMeasure.' */
  String prefix();
  
  /** @return configuration parameters name, e.g. 'weightsFile' */
  String name();
  
  /** @return human readable description of parameters */
  String description();
  
  /** @return value class type */
  Class<T> type();
  
  /**
   * @param stringValue
   *          value string representation
   */
  void setStringValue(String stringValue);
  
  /**
   * @return value string representation of current value
   */
  String getStringValue();
  
  /**
   * @param value
   *          new parameters value
   */
  void set(T value);
  
  /** @return current parameters value */
  T get();
  
  /** @return value used if not set by consumer */
  String defaultValue();
}
