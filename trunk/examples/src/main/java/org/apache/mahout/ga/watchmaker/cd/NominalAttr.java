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

import com.google.common.base.Preconditions;

class NominalAttr implements Attribute {
  
  private final String[] values;
  
  NominalAttr(String[] values) {
    Preconditions.checkArgument(values.length > 0, "values is empty");
    this.values = values;
  }
  
  public int getNbvalues() {
    return values.length;
  }
  
  @Override
  public boolean isNumerical() {
    return false;
  }
  
  /**
   * Converts a string value of a nominal attribute to an {@code int}.
   *
   * @param value
   * @return an {@code int} representing the value
   * @throws IllegalArgumentException if the value is not found.
   */
  public int valueIndex(String value) {
    for (int index = 0; index < values.length; index++) {
      if (values[index].equals(value)) {
        return index;
      }
    }
    throw new IllegalArgumentException("Value (" + value + ") not found");
  }
  
}
