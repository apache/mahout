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

package org.apache.mahout.ga.watchmaker.cd.tool;

/**
 * Used as a configuration Parameters for the mapper-combiner-reducer<br>
 * CDTOOL.ATTRIBUTES : char[] description of the attributes.<br>
 * for each attribute, takes one of this values:<br>
 * <ul>
 * <li>N : numerical attribute</li>
 * <li>C : categorical (nominal) attribute</li>
 * <li>L : label (nominal) attribute</li>
 * <li>I : ignored attribute</li>
 * </ul>
 * 
 */
public class Descriptors {

  private final char[] descriptors;

  public Descriptors(char[] descriptors) {
    assert descriptors != null && descriptors.length > 0;

    this.descriptors = descriptors;

    // check that all the descriptors are correct ('I', 'N' 'L' or 'C')
    for (int index = 0; index < descriptors.length; index++) {
      if (!isIgnored(index) && !isNumerical(index) && !isNominal(index))
        throw new RuntimeException("Bad descriptor value : "
            + descriptors[index]);
    }
  }

  public boolean isIgnored(int index) {
    return descriptors[index] == 'i' || descriptors[index] == 'I';
  }

  public boolean isNumerical(int index) {
    return descriptors[index] == 'n' || descriptors[index] == 'N';
  }

  public boolean isNominal(int index) {
    return descriptors[index] == 'c' || descriptors[index] == 'C' || isLabel(index);
  }

  public boolean isLabel(int index) {
    return descriptors[index] == 'l' || descriptors[index] == 'L';
  }
  
  public int size() {
    return descriptors.length;
  }
  
  public char[] getChars() {
    return descriptors;
  }
}
