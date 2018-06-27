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

public enum ARFFType {

  NUMERIC("numeric"),
  INTEGER("integer"),
  REAL("real"),
  NOMINAL("{"),
  DATE("date"),
  STRING("string");
  
  private final String indicator;
  
  ARFFType(String indicator) {
    this.indicator = indicator;
  }
  
  public String getIndicator() {
    return indicator;
  }
  
  public String getLabel(String line) {
    int idx = line.lastIndexOf(indicator);
    return removeQuotes(line.substring(ARFFModel.ATTRIBUTE.length(), idx));
  }

  /**
   * Remove quotes and leading/trailing whitespace from a single or double quoted string
   * @param str quotes from
   * @return  A string without quotes
   */
  public static String removeQuotes(String str) {
    String cleaned = str;
    if (cleaned != null) {
      cleaned = cleaned.trim();
      boolean isQuoted = cleaned.length() > 1
          && (cleaned.startsWith("\"") &&  cleaned.endsWith("\"")
          || cleaned.startsWith("'") &&  cleaned.endsWith("'"));
      if (isQuoted) {
        cleaned = cleaned.substring(1, cleaned.length() - 1);
      }
    }
    return cleaned;
  }
}
