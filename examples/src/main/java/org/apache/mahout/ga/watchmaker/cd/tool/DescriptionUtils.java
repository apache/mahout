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

package org.apache.mahout.ga.watchmaker.cd.tool;

import java.util.Collection;
import java.util.Collections;
import java.util.regex.Pattern;

/**
 * Utility functions to handle Attribute's description strings.
 */
public final class DescriptionUtils {

  private static final Pattern COMMA = Pattern.compile(",");

  private DescriptionUtils() {
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
      if (++ind < values.size()) {
        buffer.append(',');
      }
    }
    
    return buffer.toString();
  }
  
  public static double[] extractNumericalRange(CharSequence description) {
    String[] tokens = COMMA.split(description);
    double min = Double.parseDouble(tokens[0]);
    double max = Double.parseDouble(tokens[1]);
    return new double[] {min,max};
  }
  /**
   * Extract all available values from the description.
   * 
   * @param description
   * @param target the extracted values will be added to this collection. It
   *        will not be cleared.
   */
  public static void extractNominalValues(CharSequence description,
                                          Collection<String> target) {
    Collections.addAll(target, COMMA.split(description));
  }
  
}
