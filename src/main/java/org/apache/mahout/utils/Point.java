/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.utils;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.regex.Pattern;

public class Point {
  /**
   * Split pattern for {@link #decodePoint(String)}.
   */
  private final static Pattern splitPattern = Pattern.compile("[,]");

  /**
   * Format the point for input to a Mapper or Reducer
   *
   * @param point a Float[]
   * @return a String
   */
  public static String formatPoint(Float[] point) {
    if (point.length == 0) {
      return "[]";
    }

    final StringBuilder out = new StringBuilder();
    out.append('[');
    for (int i = 0; i < point.length; i++) {
      if (i > 0) out.append(", ");
      out.append(point[i]);
    }
    out.append(']');
    return out.toString();
  }

  /**
   * Decodes a point from its string representation.
   *
   * @param formattedString a comma-terminated String of the form
   *                        "[v1,v2,...,vn]"
   * @return the Float[] defining an n-dimensional point
   */
  public static Float[] decodePoint(String formattedString) {
    if (formattedString.charAt(0) != '[' 
      || formattedString.charAt(formattedString.length() - 1) != ']') {
      throw new IllegalArgumentException(formattedString);
    }
    formattedString = formattedString.substring(1, formattedString.length() - 1);

    final String[] pts = splitPattern.split(formattedString);
    final Float[] point = new Float[pts.length];
    for (int i = 0; i < point.length; i++) {
      point[i] = new Float(pts[i]);
    }
    return point;
  }

  /**
   * Returns a print string for the point
   *
   * @param out a String to append to
   * @param pt  the Float[] point
   * @return
   */
  public static String ptOut(String out, Float[] pt) {
    return out + formatPoint(pt);
  }

  /**
   * Return a point with length dimensions and zero values
   *
   * @param length
   * @return a Float[] representing [0,0,0,...,0]
   */
  public static Float[] origin(int length) {
    Float[] result = new Float[length];
    for (int i = 0; i < length; i++)
      result[i] = new Float(0);
    return result;
  }

  /**
   * Return the sum of the two points
   *
   * @param pt1 a Float[] point
   * @param pt2 a Float[] point
   * @return
   */
  public static Float[] sum(Float[] pt1, Float[] pt2) {
    Float[] result = pt1.clone();
    for (int i = 0; i < pt1.length; i++)
      result[i] += pt2[i];
    return result;
  }

  public static void writePointsToFile(List<Float[]> points, String fileName)
          throws IOException {
    writePointsToFileWithPayload(points, fileName, "");
  }

  public static void writePointsToFileWithPayload(List<Float[]> points,
                                                  String fileName, String payload) throws IOException {
    BufferedWriter output = new BufferedWriter(new FileWriter(fileName));
    for (Float[] point : points) {
      output.write(org.apache.mahout.utils.Point.formatPoint(point));
      output.write(payload);
      output.write("\n");
    }
    output.flush();
    output.close();
  }

}