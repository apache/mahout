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

import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;

public class Point {
  /**
   * Split pattern for {@link #decodePoint(String)}.
   */
  private static final Pattern splitPattern = Pattern.compile("[,]");

  /**
   * Format the point for input to a Mapper or Reducer
   *
   * @param point a point to format
   * @return a String
   */
  public static String formatPoint(Vector point) {
    if (point.cardinality() == 0) {
      return "[]";
    }

    final StringBuilder out = new StringBuilder();
    out.append('[');
    for (int i = 0; i < point.cardinality(); i++) {
      if (i > 0) out.append(", ");
      out.append(point.get(i));
    }
    out.append(']');
    return out.toString();
  }

  /**
   * Decodes a point from its string representation.
   *
   * @param formattedString a comma-terminated String of the form
   *    "[v1,v2,...,vn]payload". Note the payload remainder: it is optional,
   *    but can be present.
   * @return the n-dimensional point
   */
  public static Vector decodePoint(String formattedString) {
    final int closingBracketIndex = formattedString.indexOf(']');
    if (formattedString.charAt(0) != '[' || closingBracketIndex < 0) {
      throw new IllegalArgumentException(formattedString);
    }

    formattedString = formattedString.substring(1, closingBracketIndex);

    final String[] pts = splitPattern.split(formattedString);
    final Vector point = new SparseVector(pts.length);
    for (int i = 0; i < point.cardinality(); i++) {
      point.set(i, Double.parseDouble(pts[i]));
    }

    return point;
  }

  /**
   * Returns a print string for the point
   *
   * @param out a String to append to
   * @param pt  the point
   * @return
   */
  public static String ptOut(String out, Vector pt) {
    return out + formatPoint(pt);
  }

  /**
   * Return a point with length dimensions and zero values
   *
   * @param length
   * @return a point representing [0,0,0,...,0]
   */
  public static Vector origin(int length) {

    Vector point = new SparseVector(length);
    point.assign(0);

    return point;
  }

  /**
   * Return the sum of the two points
   *
   * @param pt1 first point to add
   * @param pt2 second point to add
   * @return
   */
  public static Vector sum(Vector v1, Vector v2) {
    Vector sum = v1.plus(v2);
    return sum;
  }

  public static void writePointsToFile(List<Vector> points, String fileName)
          throws IOException {
    writePointsToFileWithPayload(points, fileName, "");
  }

  public static void writePointsToFileWithPayload(List<Vector> points,
                                                  String fileName, String payload) throws IOException {
    BufferedWriter output = new BufferedWriter(new FileWriter(fileName));
    for (Vector point : points) {
      output.write(org.apache.mahout.utils.Point.formatPoint(point));
      output.write(payload);
      output.write("\n");
    }
    output.flush();
    output.close();
  }

}