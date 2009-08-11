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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.transforms.PreferenceTransform;
import org.apache.mahout.cf.taste.transforms.SimilarityTransform;

import java.util.Collection;
import java.util.concurrent.Callable;

/** Abstract superclass encapsulating functionality that is common to most implementations in this package. */
abstract class AbstractSimilarity implements UserSimilarity, ItemSimilarity {

  private final DataModel dataModel;
  private PreferenceInferrer inferrer;
  private PreferenceTransform prefTransform;
  private SimilarityTransform similarityTransform;
  private boolean weighted;
  private int cachedNumItems;
  private int cachedNumUsers;
  private final RefreshHelper refreshHelper;

  /** <p>Creates a normal (unweighted) {@link AbstractSimilarity}.</p> */
  AbstractSimilarity(DataModel dataModel) throws TasteException {
    this(dataModel, Weighting.UNWEIGHTED);
  }

  /** <p>Creates a possibly weighted {@link AbstractSimilarity}.</p> */
  AbstractSimilarity(final DataModel dataModel, Weighting weighting) throws TasteException {
    if (dataModel == null) {
      throw new IllegalArgumentException("dataModel is null");
    }
    this.dataModel = dataModel;
    this.weighted = weighting == Weighting.WEIGHTED;
    this.cachedNumItems = dataModel.getNumItems();
    this.cachedNumUsers = dataModel.getNumUsers();
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        cachedNumItems = dataModel.getNumItems();
        cachedNumUsers = dataModel.getNumUsers();
        return null;
      }
    });
    this.refreshHelper.addDependency(this.dataModel);
  }

  final DataModel getDataModel() {
    return dataModel;
  }

  final PreferenceInferrer getPreferenceInferrer() {
    return inferrer;
  }

  @Override
  public final void setPreferenceInferrer(PreferenceInferrer inferrer) {
    if (inferrer == null) {
      throw new IllegalArgumentException("inferrer is null");
    }
    refreshHelper.addDependency(inferrer);
    refreshHelper.removeDependency(this.inferrer);
    this.inferrer = inferrer;
  }

  public final PreferenceTransform getPrefTransform() {
    return prefTransform;
  }

  public final void setPrefTransform(PreferenceTransform prefTransform) {
    refreshHelper.addDependency(prefTransform);
    refreshHelper.removeDependency(this.prefTransform);
    this.prefTransform = prefTransform;
  }

  public final SimilarityTransform getSimilarityTransform() {
    return similarityTransform;
  }

  public final void setSimilarityTransform(SimilarityTransform similarityTransform) {
    refreshHelper.addDependency(similarityTransform);
    refreshHelper.removeDependency(this.similarityTransform);
    this.similarityTransform = similarityTransform;
  }

  final boolean isWeighted() {
    return weighted;
  }

  /**
   * <p>Several subclasses in this package implement this method to actually compute the similarity from figures
   * computed over users or items. Note that the computations in this class "center" the data, such that X and Y's mean
   * are 0.</p>
   *
   * <p>Note that the sum of all X and Y values must then be 0. This value isn't passed down into the standard
   * similarity computations as a result.</p>
   *
   * @param n          total number of users or items
   * @param sumXY      sum of product of user/item preference values, over all items/users prefererred by both
   *                   users/items
   * @param sumX2      sum of the square of user/item preference values, over the first item/user
   * @param sumY2      sum of the square of the user/item preference values, over the second item/user
   * @param sumXYdiff2 sum of squares of differences in X and Y values
   * @return similarity value between -1.0 and 1.0, inclusive, or {@link Double#NaN} if no similarity can be computed
   *         (e.g. when no items have been rated by both uesrs
   */
  abstract double computeResult(int n, double sumXY, double sumX2, double sumY2, double sumXYdiff2);

  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {
    PreferenceArray xPrefs = dataModel.getPreferencesFromUser(userID1);
    PreferenceArray yPrefs = dataModel.getPreferencesFromUser(userID2);
    int xLength = xPrefs.length();
    int yLength = yPrefs.length();

    if (xLength == 0 || yLength == 0) {
      return Double.NaN;
    }

    Preference xPref = xPrefs.get(0);
    Preference yPref = yPrefs.get(0);
    long xIndex = xPref.getItemID();
    long yIndex = yPref.getItemID();
    int xPrefIndex = 0;
    int yPrefIndex = 0;

    double sumX = 0.0;
    double sumX2 = 0.0;
    double sumY = 0.0;
    double sumY2 = 0.0;
    double sumXY = 0.0;
    double sumXYdiff2 = 0.0;
    int count = 0;

    boolean hasInferrer = inferrer != null;
    boolean hasPrefTransform = prefTransform != null;

    while (true) {
      int compare = xIndex < yIndex ? -1 : xIndex > yIndex ? 1 : 0;
      if (hasInferrer || compare == 0) {
        double x;
        double y;
        if (xIndex == yIndex) {
          // Both users expressed a preference for the item
          if (hasPrefTransform) {
            x = prefTransform.getTransformedValue(xPref);
            y = prefTransform.getTransformedValue(yPref);
          } else {
            x = xPref.getValue();
            y = yPref.getValue();
          }
        } else {
          // Only one user expressed a preference, but infer the other one's preference and tally
          // as if the other user expressed that preference
          if (compare < 0) {
            // X has a value; infer Y's
            x = hasPrefTransform ? prefTransform.getTransformedValue(xPref) : xPref.getValue();
            y = inferrer.inferPreference(userID2, xIndex);
          } else {
            // compare > 0
            // Y has a value; infer X's
            x = inferrer.inferPreference(userID1, yIndex);
            y = hasPrefTransform ? prefTransform.getTransformedValue(yPref) : yPref.getValue();
          }
        }
        sumXY += x * y;
        sumX += x;
        sumX2 += x * x;
        sumY += y;
        sumY2 += y * y;
        double diff = x - y;
        sumXYdiff2 += diff * diff;
        count++;
      }
      if (compare <= 0) {
        if (++xPrefIndex >= xLength) {
          break;
        }
        xPref = xPrefs.get(xPrefIndex);
        xIndex = xPref.getItemID();
      }
      if (compare >= 0) {
        if (++yPrefIndex >= yLength) {
          break;
        }
        yPref = yPrefs.get(yPrefIndex);
        yIndex = yPref.getItemID();
      }
    }

    // "Center" the data. If my math is correct, this'll do it.
    double n = (double) count;
    double meanX = sumX / n;
    double meanY = sumY / n;
    // double centeredSumXY = sumXY - meanY * sumX - meanX * sumY + n * meanX * meanY;
    double centeredSumXY = sumXY - meanY * sumX;
    // double centeredSumX2 = sumX2 - 2.0 * meanX * sumX + n * meanX * meanX;
    double centeredSumX2 = sumX2 - meanX * sumX;
    // double centeredSumY2 = sumY2 - 2.0 * meanY * sumY + n * meanY * meanY;
    double centeredSumY2 = sumY2 - meanY * sumY;

    double result = computeResult(count, centeredSumXY, centeredSumX2, centeredSumY2, sumXYdiff2);

    if (similarityTransform != null) {
      result = similarityTransform.transformSimilarity(userID1, userID2, result);
    }

    if (!Double.isNaN(result)) {
      result = normalizeWeightResult(result, count, cachedNumItems);
    }
    return result;
  }

  @Override
  public final double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    PreferenceArray xPrefs = dataModel.getPreferencesForItem(itemID1);
    PreferenceArray yPrefs = dataModel.getPreferencesForItem(itemID2);
    int xLength = xPrefs.length();
    int yLength = yPrefs.length();

    if (xLength == 0 || yLength == 0) {
      return Double.NaN;
    }

    Preference xPref = xPrefs.get(0);
    Preference yPref = yPrefs.get(0);
    long xIndex = xPref.getUserID();
    long yIndex = yPref.getUserID();
    int xPrefIndex = 1;
    int yPrefIndex = 1;

    double sumX = 0.0;
    double sumX2 = 0.0;
    double sumY = 0.0;
    double sumY2 = 0.0;
    double sumXY = 0.0;
    double sumXYdiff2 = 0.0;
    int count = 0;

    // No, pref inferrers and transforms don't appy here. I think.

    while (true) {
      int compare = xIndex < yIndex ? -1 : xIndex > yIndex ? 1 : 0;
      if (compare == 0) {
        // Both users expressed a preference for the item
        double x = xPref.getValue();
        double y = yPref.getValue();
        sumXY += x * y;
        sumX += x;
        sumX2 += x * x;
        sumY += y;
        sumY2 += y * y;
        double diff = x - y;
        sumXYdiff2 += diff * diff;
        count++;
      }
      if (compare <= 0) {
        if (xPrefIndex == xLength) {
          break;
        }
        xPref = xPrefs.get(xPrefIndex++);
        xIndex = xPref.getUserID();
      }
      if (compare >= 0) {
        if (yPrefIndex == yLength) {
          break;
        }
        yPref = yPrefs.get(yPrefIndex++);
        yIndex = yPref.getUserID();
      }
    }

    // See comments above on these computations
    double n = (double) count;
    double meanX = sumX / n;
    double meanY = sumY / n;
    // double centeredSumXY = sumXY - meanY * sumX - meanX * sumY + n * meanX * meanY;
    double centeredSumXY = sumXY - meanY * sumX;
    // double centeredSumX2 = sumX2 - 2.0 * meanX * sumX + n * meanX * meanX;
    double centeredSumX2 = sumX2 - meanX * sumX;
    // double centeredSumY2 = sumY2 - 2.0 * meanY * sumY + n * meanY * meanY;
    double centeredSumY2 = sumY2 - meanY * sumY;

    double result = computeResult(count, centeredSumXY, centeredSumX2, centeredSumY2, sumXYdiff2);

    if (similarityTransform != null) {
      result = similarityTransform.transformSimilarity(itemID1, itemID2, result);
    }

    if (!Double.isNaN(result)) {
      result = normalizeWeightResult(result, count, cachedNumUsers);
    }
    return result;
  }

  final double normalizeWeightResult(double result, int count, int num) {
    if (weighted) {
      double scaleFactor = 1.0 - (double) count / (double) (num + 1);
      if (result < 0.0) {
        result = -1.0 + scaleFactor * (1.0 + result);
      } else {
        result = 1.0 - scaleFactor * (1.0 - result);
      }
    }
    // Make sure the result is not accidentally a little outside [-1.0, 1.0] due to rounding:
    if (result < -1.0) {
      result = -1.0;
    } else if (result > 1.0) {
      result = 1.0;
    }
    return result;
  }

  @Override
  public final void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  @Override
  public final String toString() {
    return this.getClass().getSimpleName() + "[dataModel:" + dataModel + ",inferrer:" + inferrer + ']';
  }

}
