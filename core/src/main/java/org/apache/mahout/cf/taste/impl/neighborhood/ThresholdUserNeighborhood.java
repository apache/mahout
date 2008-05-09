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

package org.apache.mahout.cf.taste.impl.neighborhood;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.impl.common.SoftCache;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.User;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * <p>Computes a neigbhorhood consisting of all {@link User}s whose similarity to the
 * given {@link User} meets or exceeds a certain threshold. Similartiy is defined by the given
 * {@link UserCorrelation}.</p>
 */
public final class ThresholdUserNeighborhood extends AbstractUserNeighborhood {

  private static final Logger log = Logger.getLogger(ThresholdUserNeighborhood.class.getName());

  private final SoftCache<Object, Collection<User>> cache;

  /**
   * @param threshold similarity threshold
   * @param userCorrelation similarity metric
   * @param dataModel data model
   * @throws IllegalArgumentException if threshold is {@link Double#NaN},
   * or if samplingRate is not positive and less than or equal to 1.0, or if userCorrelation
   * or dataModel are <code>null</code>
   */
  public ThresholdUserNeighborhood(double threshold,
                                   UserCorrelation userCorrelation,
                                   DataModel dataModel) throws TasteException {
    this(threshold, userCorrelation, dataModel, 1.0);
  }

  /**
   * @param threshold similarity threshold
   * @param userCorrelation similarity metric
   * @param dataModel data model
   * @param samplingRate percentage of users to consider when building neighborhood -- decrease to
   * trade quality for performance
   * @throws IllegalArgumentException if threshold or samplingRate is {@link Double#NaN},
   * or if samplingRate is not positive and less than or equal to 1.0, or if userCorrelation
   * or dataModel are <code>null</code>
   */
  public ThresholdUserNeighborhood(double threshold,
                                   UserCorrelation userCorrelation,
                                   DataModel dataModel,
                                   double samplingRate) throws TasteException {
    super(userCorrelation, dataModel, samplingRate);
    if (Double.isNaN(threshold)) {
      throw new IllegalArgumentException("threshold must not be NaN");
    }
    this.cache = new SoftCache<Object, Collection<User>>(new Retriever(threshold), dataModel.getNumUsers());
  }

  public Collection<User> getUserNeighborhood(Object userID) throws TasteException {
    return cache.get(userID);
  }

  @Override
  public String toString() {
    return "ThresholdUserNeighborhood";
  }


  private final class Retriever implements SoftCache.Retriever<Object, Collection<User>> {

    private final double threshold;

    private Retriever(double threshold) {
      this.threshold = threshold;
    }

    public Collection<User> getValue(Object key) throws TasteException {
      if (log.isLoggable(Level.FINER)) {
        log.fine("Computing neighborhood around user ID '" + key + '\'');
      }

      DataModel dataModel = getDataModel();
      User theUser = dataModel.getUser(key);
      List<User> neighborhood = new ArrayList<User>();
      Iterator<? extends User> users = dataModel.getUsers().iterator();
      UserCorrelation userCorrelationImpl = getUserCorrelation();

      while (users.hasNext()) {
        User user = users.next();
        if (sampleForUser() && !key.equals(user.getID())) {
          double theCorrelation = userCorrelationImpl.userCorrelation(theUser, user);
          if (!Double.isNaN(theCorrelation) && theCorrelation >= threshold) {
            neighborhood.add(user);
          }
        }
      }

      if (log.isLoggable(Level.FINER)) {
        log.fine("UserNeighborhood around user ID '" + key + "' is: " + neighborhood);
      }

      return Collections.unmodifiableList(neighborhood);
    }
  }

}
