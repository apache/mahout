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

package org.apache.mahout.classifier.df.data.conditions;

import org.apache.mahout.classifier.df.data.Instance;

/**
 * True if a given attribute has a value "greater or equal" than a given value
 */
public class GreaterOrEquals extends Condition {
  
  private final int attr;
  
  private final double value;
  
  public GreaterOrEquals(int attr, double value) {
    this.attr = attr;
    this.value = value;
  }
  
  @Override
  public boolean isTrueFor(Instance v) {
    return v.get(attr) >= value;
  }
  
}
