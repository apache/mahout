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

package org.apache.mahout.classifier.df.data;

import org.apache.mahout.math.Vector;

/**
 * Represents one data instance.
 */
public class Instance {
  
  /** attributes, except LABEL and IGNORED */
  private final Vector attrs;
  
  public Instance(Vector attrs) {
    this.attrs = attrs;
  }
  
  /**
   * Return the attribute at the specified position
   * 
   * @param index
   *          position of the attribute to retrieve
   * @return value of the attribute
   */
  public double get(int index) {
    return attrs.getQuick(index);
  }
  
  /**
   * Set the value at the given index
   * 
   * @param value
   *          a double value to set
   */
  public void set(int index, double value) {
    attrs.set(index, value);
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof Instance)) {
      return false;
    }
    
    Instance instance = (Instance) obj;
    
    return /*id == instance.id &&*/ attrs.equals(instance.attrs);
    
  }
  
  @Override
  public int hashCode() {
    return /*id +*/ attrs.hashCode();
  }
}
