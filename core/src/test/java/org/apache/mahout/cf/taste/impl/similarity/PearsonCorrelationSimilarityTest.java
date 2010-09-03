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

import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.junit.Test;

/** <p>Tests {@link PearsonCorrelationSimilarity}.</p> */
public final class PearsonCorrelationSimilarityTest extends SimilarityTestCase {

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
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

  @Test
  public void testRefresh() throws Exception {
    // Make sure this doesn't throw an exception
    new PearsonCorrelationSimilarity(getDataModel()).refresh(null);
  }

  @Test
  public void testInferrer() throws Exception {
    DataModel dataModel = getDataModel(
      new long[] {1, 2},
      new Double[][] {
              {null, 1.0, 2.0,  null, null, 6.0},
              {1.0, 8.0, null, 3.0,  4.0,  null},
      });
    UserSimilarity similarity = new PearsonCorrelationSimilarity(dataModel);
    similarity.setPreferenceInferrer(new PreferenceInferrer() {
      @Override
      public float inferPreference(long userID, long itemID) {
        return 1.0f;
      }
      @Override
      public void refresh(Collection<Refreshable> alreadyRefreshed) {
      }
    });

    assertEquals(-0.435285750066007, similarity.userSimilarity(1L, 2L), EPSILON);
  }

}
