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

package org.apache.mahout.common;

import java.util.regex.Pattern;

import com.thoughtworks.xstream.XStream;

/**
 * Offers two methods to convert an object to a string representation and restore the object given its string
 * representation. Should use Hadoop Stringifier whenever available.
 */
public final class StringUtils {
  
  private static final XStream XSTREAM = new XStream();
  private static final Pattern NEWLINE_PATTERN = Pattern.compile("\n");
  private static final Pattern XMLRESERVED = Pattern.compile("\"|\\&|\\<|\\>|\'");

  private StringUtils() {
  // do nothing
  }
  
  /**
   * Converts the object to a one-line string representation
   * 
   * @param obj
   *          the object to convert
   * @return the string representation of the object
   */
  public static String toString(Object obj) {
    return NEWLINE_PATTERN.matcher(XSTREAM.toXML(obj)).replaceAll("");
  }
  
  /**
   * Restores the object from its string representation.
   * 
   * @param str
   *          the string representation of the object
   * @return restored object
   */
  public static <T> T fromString(String str) {
    return (T) XSTREAM.fromXML(str);
  }

  public static String escapeXML(CharSequence input) {
    return XMLRESERVED.matcher(input).replaceAll("_");
  }
}
