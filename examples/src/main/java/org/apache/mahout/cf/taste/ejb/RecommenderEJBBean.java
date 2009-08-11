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

package org.apache.mahout.cf.taste.ejb;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.recommender.Rescorer;

import javax.ejb.CreateException;
import javax.ejb.SessionBean;
import javax.ejb.SessionContext;
import javax.naming.Context;
import javax.naming.InitialContext;
import javax.naming.NamingException;
import java.util.Collection;
import java.util.List;

/**
 * <p>Recommender EJB bean implementation.</p>
 *
 * <p>This class exposes a subset of the {@link Recommender} API. In particular it
 * does not support {@link Recommender#getDataModel()}
 * since it doesn't make sense to access this via an EJB component.</p>
 */
public class RecommenderEJBBean implements SessionBean {

  private Recommender recommender;

  public List<RecommendedItem> recommend(long userID, int howMany) throws TasteException {
    return recommender.recommend(userID, howMany);
  }

  public List<RecommendedItem> recommend(long userID, int howMany, Rescorer<Long> rescorer)
          throws TasteException {
    return recommender.recommend(userID, howMany, rescorer);
  }


  public double estimatePreference(long userID, long itemID) throws TasteException {
    return recommender.estimatePreference(userID, itemID);
  }

  public void setPreference(long userID, long itemID, float value) throws TasteException {
    recommender.setPreference(userID, itemID, value);
  }

  public void removePreference(long userID, long itemID) throws TasteException {
    recommender.removePreference(userID, itemID);
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    recommender.refresh(alreadyRefreshed);
  }

  @Override
  public void setSessionContext(SessionContext sessionContext) {
    // Do nothing
  }

  public void ejbCreate() throws CreateException {
    Context ctx = null;
    try {

      ctx = new InitialContext();
      String recommenderClassName = (String) ctx.lookup("java:comp/env/recommender-class");
      if (recommenderClassName == null) {
        String recommenderJNDIName = (String) ctx.lookup("java:comp/env/recommender-jndi-name");
        if (recommenderJNDIName == null) {
          throw new CreateException("recommender-class and recommender-jndi-name env-entry not defined");
        }
        recommender = (Recommender) ctx.lookup("java:comp/env/" + recommenderJNDIName);
      } else {
        recommender = Class.forName(recommenderClassName).asSubclass(Recommender.class).newInstance();
      }

    } catch (NamingException ne) {
      throw new CreateException(ne.toString());
    } catch (ClassNotFoundException cnfe) {
      throw new CreateException(cnfe.toString());
    } catch (InstantiationException ie) {
      throw new CreateException(ie.toString());
    } catch (IllegalAccessException iae) {
      throw new CreateException(iae.toString());
    } finally {
      if (ctx != null) {
        try {
          ctx.close();
        } catch (NamingException ne) {
          throw new CreateException(ne.toString());
        }
      }
    }
  }

  @Override
  public void ejbRemove() {
    // Do nothing
  }

  @Override
  public void ejbActivate() {
    // Do nothing: stateless session beans are not passivated/activated
  }

  @Override
  public void ejbPassivate() {
    // Do nothing: stateless session beans are not passivated/activated
  }

  @Override
  public String toString() {
    return "RecommenderEJBBean[recommender:" + recommender + ']';
  }

}
