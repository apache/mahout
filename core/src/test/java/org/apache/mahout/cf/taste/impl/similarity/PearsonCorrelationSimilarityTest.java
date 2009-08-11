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

package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

/** <p>Tests {@link PearsonCorrelationSimilarity}.</p> */
public final class PearsonCorrelationSimilarityTest extends SimilarityTestCase {

  public void testFullCorrelation1() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {3.0, -2.0},
                    {3.0, -2.0},
            });
    double correlation = new PearsonCorrelationSimilarity(dataModel).userSimilarity(1, 2);
    assertCorrelationEquals(1.0, correlation);
  }

  public void testFullCorrelation1Weighted() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {3.0, -2.0},
                    {3.0, -2.0},
            });
    double correlation = new PearsonCorrelationSimilarity(dataModel, Weighting.WEIGHTED).userSimilarity(1, 2);
    assertCorrelationEquals(1.0, correlation);
  }

  public void testFullCorrelation2() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {3.0, 3.0},
                    {3.0, 3.0},
            });
    double correlation = new PearsonCorrelationSimilarity(dataModel).userSimilarity(1, 2);
    // Yeah, undefined in this case
    assertTrue(Double.isNaN(correlation));
  }

  public void testNoCorrelation1() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {3.0, -2.0},
                    {-3.0, 2.0},
            });
    double correlation = new PearsonCorrelationSimilarity(dataModel).userSimilarity(1, 2);
    assertCorrelationEquals(-1.0, correlation);
  }

  public void testNoCorrelation1Weighted() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {3.0, -2.0},
                    {-3.0, 2.0},
            });
    double correlation = new PearsonCorrelationSimilarity(dataModel, Weighting.WEIGHTED).userSimilarity(1, 2);
    assertCorrelationEquals(-1.0, correlation);
  }

  public void testNoCorrelation2() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {null, 1.0, null},
                    {null, null, 1.0},
            });
    double correlation = new PearsonCorrelationSimilarity(dataModel).userSimilarity(1, 2);
    assertTrue(Double.isNaN(correlation));
  }

  public void testNoCorrelation3() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {90.0, 80.0, 70.0},
                    {70.0, 80.0, 90.0},
            });
    double correlation = new PearsonCorrelationSimilarity(dataModel).userSimilarity(1, 2);
    assertCorrelationEquals(-1.0, correlation);
  }

  public void testSimple() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {1.0, 2.0, 3.0},
                    {2.0, 5.0, 6.0},
            });
    double correlation = new PearsonCorrelationSimilarity(dataModel).userSimilarity(1, 2);
    assertCorrelationEquals(0.9607689228305227, correlation);
  }

  public void testSimpleWeighted() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {1.0, 2.0, 3.0},
                    {2.0, 5.0, 6.0},
            });
    double correlation = new PearsonCorrelationSimilarity(dataModel, Weighting.WEIGHTED).userSimilarity(1, 2);
    assertCorrelationEquals(0.9901922307076306, correlation);
  }

  public void testFullItemCorrelation1() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {3.0, 3.0},
                    {-2.0, -2.0},
            });
    double correlation =
        new PearsonCorrelationSimilarity(dataModel).itemSimilarity(0, 1);
    assertCorrelationEquals(1.0, correlation);
  }

  public void testFullItemCorrelation2() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {3.0, 3.0},
                    {3.0, 3.0},
            });
    double correlation =
        new PearsonCorrelationSimilarity(dataModel).itemSimilarity(0, 1);
    // Yeah, undefined in this case
    assertTrue(Double.isNaN(correlation));
  }

  public void testNoItemCorrelation1() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {3.0, -3.0},
                    {2.0, -2.0},
            });
    double correlation =
        new PearsonCorrelationSimilarity(dataModel).itemSimilarity(0, 1);
    assertCorrelationEquals(-1.0, correlation);
  }

  public void testNoItemCorrelation2() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2},
            new Double[][] {
                    {null, 1.0, null},
                    {null, null, 1.0},
            });
    double correlation =
        new PearsonCorrelationSimilarity(dataModel).itemSimilarity(1, 2);
    assertTrue(Double.isNaN(correlation));
  }

  public void testNoItemCorrelation3() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2, 3},
            new Double[][] {
                    {90.0, 70.0},
                    {80.0, 80.0},
                    {70.0, 90.0},
            });
    double correlation =
        new PearsonCorrelationSimilarity(dataModel).itemSimilarity(0, 1);
    assertCorrelationEquals(-1.0, correlation);
  }

  public void testSimpleItem() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2, 3},
            new Double[][] {
                    {1.0, 2.0},
                    {2.0, 5.0},
                    {3.0, 6.0},
            });
    double correlation =
        new PearsonCorrelationSimilarity(dataModel).itemSimilarity(0, 1);
    assertCorrelationEquals(0.9607689228305227, correlation);
  }

  public void testSimpleItemWeighted() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2, 3},
            new Double[][] {
                    {1.0, 2.0},
                    {2.0, 5.0},
                    {3.0, 6.0},
            });
    ItemSimilarity itemSimilarity = new PearsonCorrelationSimilarity(dataModel, Weighting.WEIGHTED);
    double correlation = itemSimilarity.itemSimilarity(0, 1);
    assertCorrelationEquals(0.9901922307076306, correlation);
  }

  public void testRefresh() throws Exception {
    // Make sure this doesn't throw an exception
    new PearsonCorrelationSimilarity(getDataModel()).refresh(null);
  }

}
