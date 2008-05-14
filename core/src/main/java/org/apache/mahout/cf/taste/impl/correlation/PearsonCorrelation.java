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

package org.apache.mahout.cf.taste.impl.correlation;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.correlation.ItemCorrelation;
import org.apache.mahout.cf.taste.correlation.PreferenceInferrer;
import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.transforms.CorrelationTransform;
import org.apache.mahout.cf.taste.transforms.PreferenceTransform2;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>An implementation of the Pearson correlation. For {@link User}s X and Y, the following values
 * are calculated:</p>
 *
 * <ul>
 * <li>sumX2: sum of the square of all X's preference values</li>
 * <li>sumY2: sum of the square of all Y's preference values</li>
 * <li>sumXY: sum of the product of X and Y's preference value for all items for which both
 * X and Y express a preference</li>
 * </ul>
 *
 * <p>The correlation is then:
 *
 * <p><code>sumXY / sqrt(sumX2 * sumY2)</code></p>
 *
 * <p>where <code>size</code> is the number of {@link Item}s in the {@link DataModel}.</p>
 *
 * <p>Note that this correlation "centers" its data, shifts the user's preference values so that
 * each of their means is 0. This is necessary to achieve expected behavior on all data sets.</p>
 *
 * <p>This correlation implementation is equivalent to the cosine measure correlation since the data it
 * receives is assumed to be centered -- mean is 0. The correlation may be interpreted as the cosine of the
 * angle between the two vectors defined by the users' preference values.</p>
 */
public final class PearsonCorrelation implements UserCorrelation, ItemCorrelation {

  private static final Logger log = LoggerFactory.getLogger(PearsonCorrelation.class);

  private final DataModel dataModel;
  private PreferenceInferrer inferrer;
  private PreferenceTransform2 prefTransform;
  private CorrelationTransform<Object> correlationTransform;
  private boolean weighted;

  /**
   * <p>Creates a normal (unweighted) {@link PearsonCorrelation}.</p>
   *
   * @param dataModel
   */
  public PearsonCorrelation(DataModel dataModel) {
    this(dataModel, false);
  }

  /**
   * <p>Creates a weighted {@link PearsonCorrelation}.</p>
   *
   * @param dataModel
   * @param weighted
   */
  public PearsonCorrelation(DataModel dataModel, boolean weighted) {
    if (dataModel == null) {
      throw new IllegalArgumentException("dataModel is null");
    }
    this.dataModel = dataModel;
    this.weighted = weighted;
  }

  /**
   * <p>Several subclasses in this package implement this method to actually compute the correlation
   * from figures computed over users or items. Note that the computations in this class "center" the
   * data, such that X and Y's mean are 0.</p>
   *
   * <p>Note that the sum of all X and Y values must then be 0. This value isn't passed down into
   * the standard correlation computations as a result.</p>
   *
   * @param n total number of users or items
   * @param sumXY sum of product of user/item preference values, over all items/users prefererred by
   * both users/items
   * @param sumX2 sum of the square of user/item preference values, over the first item/user
   * @param sumY2 sum of the square of the user/item preference values, over the second item/user
   * @return correlation value between -1.0 and 1.0, inclusive, or {@link Double#NaN} if no correlation
   *         can be computed (e.g. when no {@link Item}s have been rated by both {@link User}s
   */
  private static double computeResult(int n, double sumXY, double sumX2, double sumY2) {
    if (n == 0) {
      return Double.NaN;
    }
    // Note that sum of X and sum of Y don't appear here since they are assumed to be 0;
    // the data is assumed to be centered.
    double xTerm = Math.sqrt(sumX2);
    double yTerm = Math.sqrt(sumY2);
    double denominator = xTerm * yTerm;
    if (denominator == 0.0) {
      // One or both parties has -all- the same ratings;
      // can't really say much correlation under this measure
      return Double.NaN;
    }
    return sumXY / denominator;
  }

  DataModel getDataModel() {
    return dataModel;
  }

  PreferenceInferrer getPreferenceInferrer() {
    return inferrer;
  }

  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    if (inferrer == null) {
      throw new IllegalArgumentException("inferrer is null");
    }
    this.inferrer = inferrer;
  }

  public PreferenceTransform2 getPrefTransform() {
    return prefTransform;
  }

  public void setPrefTransform(PreferenceTransform2 prefTransform) {
    this.prefTransform = prefTransform;
  }

  public CorrelationTransform<?> getCorrelationTransform() {
    return correlationTransform;
  }

  public void setCorrelationTransform(CorrelationTransform<Object> correlationTransform) {
    this.correlationTransform = correlationTransform;
  }

  boolean isWeighted() {
    return weighted;
  }

  public double userCorrelation(User user1, User user2) throws TasteException {

    if (user1 == null || user2 == null) {
      throw new IllegalArgumentException("user1 or user2 is null");
    }

    Preference[] xPrefs = user1.getPreferencesAsArray();
    Preference[] yPrefs = user2.getPreferencesAsArray();

    if (xPrefs.length == 0 || yPrefs.length == 0) {
      return Double.NaN;
    }

    Preference xPref = xPrefs[0];
    Preference yPref = yPrefs[0];
    Item xIndex = xPref.getItem();
    Item yIndex = yPref.getItem();
    int xPrefIndex = 1;
    int yPrefIndex = 1;

    double sumX = 0.0;
    double sumX2 = 0.0;
    double sumY = 0.0;
    double sumY2 = 0.0;
    double sumXY = 0.0;
    int count = 0;

    boolean hasInferrer = inferrer != null;
    boolean hasPrefTransform = prefTransform != null;

    while (true) {
      int compare = xIndex.compareTo(yIndex);
      if (hasInferrer || compare == 0) {
        double x;
        double y;
        if (compare == 0) {
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
            if (hasPrefTransform) {
              x = prefTransform.getTransformedValue(xPref);
            } else {
              x = xPref.getValue();
            }
            y = inferrer.inferPreference(user2, xIndex);
          } else {
            // compare > 0
            // Y has a value; infer X's
            x = inferrer.inferPreference(user1, yIndex);
            if (hasPrefTransform) {
              y = prefTransform.getTransformedValue(yPref);
            } else {
              y = yPref.getValue();
            }
          }
        }
        sumXY += x * y;
        sumX += x;
        sumX2 += x * x;
        sumY += y;
        sumY2 += y * y;
        count++;
      }
      if (compare <= 0) {
        if (xPrefIndex == xPrefs.length) {
          break;
        }
        xPref = xPrefs[xPrefIndex++];
        xIndex = xPref.getItem();
      }
      if (compare >= 0) {
        if (yPrefIndex == yPrefs.length) {
          break;
        }
        yPref = yPrefs[yPrefIndex++];
        yIndex = yPref.getItem();
      }
    }

    // "Center" the data. If my math is correct, this'll do it.
    double n = (double) count;
    double meanX = sumX / n;
    double meanY = sumY / n;
    double centeredSumXY = sumXY - meanY * sumX - meanX * sumY + n * meanX * meanY;
    double centeredSumX2 = sumX2 - 2.0 * meanX * sumX + n * meanX * meanX;
    double centeredSumY2 = sumY2 - 2.0 * meanY * sumY + n * meanY * meanY;

    double result = computeResult(count, centeredSumXY, centeredSumX2, centeredSumY2);

    if (correlationTransform != null) {
      result = correlationTransform.transformCorrelation(user1, user2, result);
    }

    if (!Double.isNaN(result)) {
      result = normalizeWeightResult(result, count, dataModel.getNumItems());
    }

    if (log.isTraceEnabled()) {
      log.trace("UserCorrelation between " + user1 + " and " + user2 + " is " + result);
    }
    return result;
  }

  public double itemCorrelation(Item item1, Item item2) throws TasteException {

    if (item1 == null || item2 == null) {
      throw new IllegalArgumentException("item1 or item2 is null");
    }

    Preference[] xPrefs = dataModel.getPreferencesForItemAsArray(item1.getID());
    Preference[] yPrefs = dataModel.getPreferencesForItemAsArray(item2.getID());

    if (xPrefs.length == 0 || yPrefs.length == 0) {
      return Double.NaN;
    }

    Preference xPref = xPrefs[0];
    Preference yPref = yPrefs[0];
    User xIndex = xPref.getUser();
    User yIndex = yPref.getUser();
    int xPrefIndex = 1;
    int yPrefIndex = 1;

    double sumX = 0.0;
    double sumX2 = 0.0;
    double sumY = 0.0;
    double sumY2 = 0.0;
    double sumXY = 0.0;
    int count = 0;

    // No, pref inferrers and transforms don't appy here. I think.

    while (true) {
      int compare = xIndex.compareTo(yIndex);
      if (compare == 0) {
        // Both users expressed a preference for the item
        double x = xPref.getValue();
        double y = yPref.getValue();
        sumXY += x * y;
        sumX += x;
        sumX2 += x * x;
        sumY += y;
        sumY2 += y * y;
        count++;
      }
      if (compare <= 0) {
        if (xPrefIndex == xPrefs.length) {
          break;
        }
        xPref = xPrefs[xPrefIndex++];
        xIndex = xPref.getUser();
      }
      if (compare >= 0) {
        if (yPrefIndex == yPrefs.length) {
          break;
        }
        yPref = yPrefs[yPrefIndex++];
        yIndex = yPref.getUser();
      }
    }

    // See comments above on these computations
    double n = (double) count;
    double meanX = sumX / n;
    double meanY = sumY / n;
    double centeredSumXY = sumXY - meanY * sumX - meanX * sumY + n * meanX * meanY;
    double centeredSumX2 = sumX2 - 2.0 * meanX * sumX + n * meanX * meanX;
    double centeredSumY2 = sumY2 - 2.0 * meanY * sumY + n * meanY * meanY;

    double result = computeResult(count, centeredSumXY, centeredSumX2, centeredSumY2);

    if (correlationTransform != null) {
      result = correlationTransform.transformCorrelation(item1, item2, result);
    }

    if (!Double.isNaN(result)) {
      result = normalizeWeightResult(result, count, dataModel.getNumUsers());
    }

    if (log.isTraceEnabled()) {
      log.trace("UserCorrelation between " + item1 + " and " + item2 + " is " + result);
    }
    return result;
  }

  private double normalizeWeightResult(double result, int count, int num) {
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

  public void refresh() {
    dataModel.refresh();
    if (inferrer != null) {
      inferrer.refresh();
    }
    if (prefTransform != null) {
      prefTransform.refresh();
    }
    if (correlationTransform != null) {
      correlationTransform.refresh();
    }
  }

  @Override
  public String toString() {
    return "PearsonCorrelation[dataModel:" + dataModel + ",inferrer:" + inferrer + ']';
  }

}
