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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

/**
 * <p>Computes a neigbhorhood consisting of the nearest n {@link User}s to a given {@link User}.
 * "Nearest" is defined by the given {@link UserCorrelation}.</p>
 */
public final class NearestNUserNeighborhood extends AbstractUserNeighborhood {

  private static final Logger log = LoggerFactory.getLogger(NearestNUserNeighborhood.class);

  private final SoftCache<Object, Collection<User>> cache;

  /**
   * @param n neighborhood size
   * @param userCorrelation nearness metric
   * @param dataModel data model
   * @throws IllegalArgumentException if n &lt; 1, or userCorrelation or dataModel are <code>null</code>
   */
  public NearestNUserNeighborhood(int n,
                                  UserCorrelation userCorrelation,
                                  DataModel dataModel) throws TasteException {
    this(n, userCorrelation, dataModel, 1.0);
  }

  /**
   * @param n neighborhood size
   * @param userCorrelation nearness metric
   * @param dataModel data model
   * @param samplingRate percentage of users to consider when building neighborhood -- decrease to
   * trade quality for performance
   * @throws IllegalArgumentException if n &lt; 1 or samplingRate is NaN or not in (0,1],
   * or userCorrelation or dataModel are <code>null</code>
   */
  public NearestNUserNeighborhood(int n,
                                  UserCorrelation userCorrelation,
                                  DataModel dataModel,
                                  double samplingRate) throws TasteException {
    super(userCorrelation, dataModel, samplingRate);
    if (n < 1) {
      throw new IllegalArgumentException("n must be at least 1");
    }
    this.cache = new SoftCache<Object, Collection<User>>(new Retriever(n), dataModel.getNumUsers());
  }

  public Collection<User> getUserNeighborhood(Object userID) throws TasteException {
    return cache.get(userID);
  }

  @Override
  public String toString() {
    return "NearestNUserNeighborhood";
  }


  private final class Retriever implements SoftCache.Retriever<Object, Collection<User>> {

    private final int n;

    private Retriever(int n) {
      this.n = n;
    }

    public Collection<User> getValue(Object key) throws TasteException {
      log.trace("Computing neighborhood around user ID '{}'", key);

      DataModel dataModel = getDataModel();
      User theUser = dataModel.getUser(key);
      UserCorrelation userCorrelationImpl = getUserCorrelation();

      LinkedList<UserCorrelationPair> queue = new LinkedList<UserCorrelationPair>();
      boolean full = false;
      for (User user : dataModel.getUsers()) {
        if (sampleForUser() && !key.equals(user.getID())) {
          double theCorrelation = userCorrelationImpl.userCorrelation(theUser, user);
          if (!Double.isNaN(theCorrelation) && (!full || theCorrelation > queue.getLast().theCorrelation)) {
            ListIterator<UserCorrelationPair> iterator = queue.listIterator(queue.size());
            while (iterator.hasPrevious()) {
              if (theCorrelation <= iterator.previous().theCorrelation) {
                iterator.next();
                break;
              }
            }
            iterator.add(new UserCorrelationPair(user, theCorrelation));
            if (full) {
              queue.removeLast();
            } else if (queue.size() > n) {
              full = true;
              queue.removeLast();
            }
          }
        }
      }

      List<User> neighborhood = new ArrayList<User>(queue.size());
      for (UserCorrelationPair pair : queue) {
        neighborhood.add(pair.user);
      }

      log.trace("UserNeighborhood around user ID '{}' is: {}", key, neighborhood);

      return Collections.unmodifiableList(neighborhood);
    }
  }

  private static final class UserCorrelationPair implements Comparable<UserCorrelationPair> {

    final User user;
    final double theCorrelation;

    private UserCorrelationPair(User user, double theCorrelation) {
      this.user = user;
      this.theCorrelation = theCorrelation;
    }

    @Override
    public int hashCode() {
      return user.hashCode() ^ Double.valueOf(theCorrelation).hashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof UserCorrelationPair)) {
        return false;
      }
      UserCorrelationPair other = (UserCorrelationPair) o;
      return user.equals(other.user) && theCorrelation == other.theCorrelation;
    }

    public int compareTo(UserCorrelationPair otherPair) {
      double otherCorrelation = otherPair.theCorrelation;
      return theCorrelation > otherCorrelation ? -1 : theCorrelation < otherCorrelation ? 1 : 0;
    }
  }

}
