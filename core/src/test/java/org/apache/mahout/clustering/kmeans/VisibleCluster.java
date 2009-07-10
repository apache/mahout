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

package org.apache.mahout.clustering.kmeans;

import org.apache.mahout.matrix.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * This Cluster subclass maintains a list of points in the cluster so it can include them in its toString method. Useful
 * for debugging but not practical for production use since it holds onto all its points.
 */
public class VisibleCluster extends Cluster {

  private List<Vector> points;

  public VisibleCluster(Vector point) {
    super(point);
    points = new ArrayList<Vector>();
  }

  @Override
  public void recomputeCenter() {
    super.recomputeCenter();
    points = new ArrayList<Vector>();
  }

  /**
   * Add a point to the canopy
   *
   * @param point a Double[]
   */
  @Override
  public void addPoint(Vector point) {
    super.addPoint(point);
    points.add(point);
  }

  /** Return a printable representation of this object, using the user supplied identifier */
  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    result.append(super.toString());
    result.append(": ");
    for (Vector pt : points) {
      result.append(pt.asFormatString());
    }
    return result.toString();
  }

}
