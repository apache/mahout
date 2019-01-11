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

package org.apache.mahout.cf.taste.impl.model.jdbc;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.model.GenericBooleanPrefDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.JDBCDataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.concurrent.Callable;

/**
 * A {@link DataModel} which loads, and can re-load, data from a JDBC-backed {@link JDBCDataModel} into memory, as a
 * {@link GenericDataModel} or {@link GenericBooleanPrefDataModel}. It is intended to provide the speed
 * advantage of in-memory representation but be able to update periodically to pull in new data from a database source.
 */
public final class ReloadFromJDBCDataModel implements DataModel {

  private static final Logger log = LoggerFactory.getLogger(ReloadFromJDBCDataModel.class);

  private DataModel delegateInMemory;
  private final JDBCDataModel delegate;
  private final RefreshHelper refreshHelper;

  public ReloadFromJDBCDataModel(JDBCDataModel delegate) throws TasteException {
    this.delegate = Preconditions.checkNotNull(delegate);
    refreshHelper = new RefreshHelper(new Callable<Void>() {
      @Override
      public Void call() {
        reload();
        return null;  //To change body of implemented methods use File | Settings | File Templates.
      }
    });
    refreshHelper.addDependency(delegate);
    reload();
    if (delegateInMemory == null) {
      throw new TasteException("Failed to load data into memory");
    }
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  private void reload() {
    try {
      // Load new in-memory representation,
      log.info("Loading new JDBC delegate data...");
      DataModel newDelegateInMemory =
          delegate.hasPreferenceValues()
          ? new GenericDataModel(delegate.exportWithPrefs())
          : new GenericBooleanPrefDataModel(delegate.exportWithIDsOnly());
      // and then swap to it.
      log.info("New data loaded.");
      delegateInMemory = newDelegateInMemory;
    } catch (TasteException te) {
      log.warn("Error while reloading JDBC delegate data", te);
      // But continue with whatever is loaded
    }
  }

  public JDBCDataModel getDelegate() {
    return delegate;
  }

  public DataModel getDelegateInMemory() {
    return delegateInMemory;
  }

  // Delegated methods:

  @Override
  public LongPrimitiveIterator getUserIDs() throws TasteException {
    return delegateInMemory.getUserIDs();
  }

  @Override
  public PreferenceArray getPreferencesFromUser(long id) throws TasteException {
    return delegateInMemory.getPreferencesFromUser(id);
  }

  @Override
  public FastIDSet getItemIDsFromUser(long id) throws TasteException {
    return delegateInMemory.getItemIDsFromUser(id);
  }

  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    return delegateInMemory.getPreferenceValue(userID, itemID);
  }

  @Override
  public Long getPreferenceTime(long userID, long itemID) throws TasteException {
    return delegateInMemory.getPreferenceTime(userID, itemID);
  }

  @Override
  public LongPrimitiveIterator getItemIDs() throws TasteException {
    return delegateInMemory.getItemIDs();
  }

  @Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
    return delegateInMemory.getPreferencesForItem(itemID);
  }

  @Override
  public int getNumItems() throws TasteException {
    return delegateInMemory.getNumItems();
  }

  @Override
  public int getNumUsers() throws TasteException {
    return delegateInMemory.getNumUsers();
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID) throws TasteException {
    return delegateInMemory.getNumUsersWithPreferenceFor(itemID);
  }

  @Override
  public int getNumUsersWithPreferenceFor(long itemID1, long itemID2) throws TasteException {
    return delegateInMemory.getNumUsersWithPreferenceFor(itemID1, itemID2);
  }

  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    delegateInMemory.setPreference(userID, itemID, value);
  }

  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    delegateInMemory.removePreference(userID, itemID);
  }

  @Override
  public boolean hasPreferenceValues() {
    return delegateInMemory.hasPreferenceValues();
  }

  @Override
  public float getMaxPreference() {
    return delegateInMemory.getMaxPreference();
  }

  @Override
  public float getMinPreference() {
    return delegateInMemory.getMinPreference();
  }

}
