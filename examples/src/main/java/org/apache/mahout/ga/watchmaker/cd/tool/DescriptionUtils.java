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

import java.util.Collection;
import java.util.StringTokenizer;

/**
 * Utility functions to handle Attribute's description strings.
 */
public class DescriptionUtils {
  private DescriptionUtils() {
  }

  public static class Range {
    public final double min;
    public final double max;
    
    public Range(double min, double max) {
      this.max = max;
      this.min = min;
    }
  }
  
  /**
   * Create a numerical attribute description.
   * 
   * @param min
   * @param max
   */
  public static String createNumericalDescription(double min, double max) {
    return min + "," + max;
  }

  /**
   * Create a nominal description from the possible values.
   * 
   * @param values
   */
  public static String createNominalDescription(Collection<String> values) {
    StringBuilder buffer = new StringBuilder();
    int ind = 0;

    for (String value : values) {
      buffer.append(value);
      if (++ind < values.size())
        buffer.append(',');
    }

    return buffer.toString();
  }

  public static Range extractNumericalRange(String description) {
    StringTokenizer tokenizer = new StringTokenizer(description, ",");
    double min = Double.parseDouble(tokenizer.nextToken());
    double max = Double.parseDouble(tokenizer.nextToken());
    
    return new Range(min, max);
  }
  /**
   * Extract all available values from the description.
   * 
   * @param description
   * @param target the extracted values will be added to this collection. It
   *        will not be cleared.
   */
  public static void extractNominalValues(String description,
      Collection<String> target) {
    StringTokenizer tokenizer = new StringTokenizer(description, ",");
    while (tokenizer.hasMoreTokens()) {
      target.add(tokenizer.nextToken());
    }
  }

}
