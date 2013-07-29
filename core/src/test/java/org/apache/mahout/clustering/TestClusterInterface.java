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

package org.apache.mahout.clustering;

import org.apache.mahout.clustering.canopy.Canopy;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;

import org.junit.Test;

public final class TestClusterInterface extends MahoutTestCase {

  private static final DistanceMeasure measure = new ManhattanDistanceMeasure();

  @Test
  public void testCanopyAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster cluster = new Canopy(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("C-123{n=0 c=[1.100, 2.200, 3.300] r=[]}", formatString);
  }

  @Test
  public void testCanopyAsFormatStringSparse() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Cluster cluster = new Canopy(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("C-123{n=0 c=[0:1.100, 2:3.300] r=[]}", formatString);
  }

  @Test
  public void testCanopyAsFormatStringWithBindings() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster cluster = new Canopy(m, 123, measure);
    String[] bindings = { "fee", null, null };
    String formatString = cluster.asFormatString(bindings);
    assertEquals("C-123{n=0 c=[fee:1.100, 1:2.200, 2:3.300] r=[]}", formatString);
  }

  @Test
  public void testCanopyAsFormatStringSparseWithBindings() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Cluster cluster = new Canopy(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("C-123{n=0 c=[0:1.100, 2:3.300] r=[]}", formatString);
  }

  @Test
  public void testClusterAsFormatString() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster cluster = new org.apache.mahout.clustering.kmeans.Kluster(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("CL-123{n=0 c=[1.100, 2.200, 3.300] r=[]}", formatString);
  }

  @Test
  public void testClusterAsFormatStringSparse() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Cluster cluster = new org.apache.mahout.clustering.kmeans.Kluster(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("CL-123{n=0 c=[0:1.100, 2:3.300] r=[]}", formatString);
  }

  @Test
  public void testClusterAsFormatStringWithBindings() {
    double[] d = { 1.1, 2.2, 3.3 };
    Vector m = new DenseVector(d);
    Cluster cluster = new org.apache.mahout.clustering.kmeans.Kluster(m, 123, measure);
    String[] bindings = { "fee", null, "foo" };
    String formatString = cluster.asFormatString(bindings);
    assertEquals("CL-123{n=0 c=[fee:1.100, 1:2.200, foo:3.300] r=[]}", formatString);
  }

  @Test
  public void testClusterAsFormatStringSparseWithBindings() {
    double[] d = { 1.1, 0.0, 3.3 };
    Vector m = new SequentialAccessSparseVector(3);
    m.assign(d);
    Cluster cluster = new org.apache.mahout.clustering.kmeans.Kluster(m, 123, measure);
    String formatString = cluster.asFormatString(null);
    assertEquals("CL-123{n=0 c=[0:1.100, 2:3.300] r=[]}", formatString);
  }

}
