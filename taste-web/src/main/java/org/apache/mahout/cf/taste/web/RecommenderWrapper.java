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

package org.apache.mahout.cf.taste.web;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.common.IOUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.List;

/**
 * Users of the packaging and deployment mechanism in this module need
 * to produce a {@link Recommender} implementation with a no-arg constructor,
 * which will internally build the desired {@link Recommender} and delegate
 * to it. This wrapper simplifies that process. Simply extend this class and
 * implement {@link #buildRecommender()}.
 */
public abstract class RecommenderWrapper implements Recommender {

  private final Recommender delegate;

  protected RecommenderWrapper() throws TasteException, IOException {
    this.delegate = buildRecommender();
  }

  /**
   * @return the {@link Recommender} which should be used to produce recommendations
   *  by this wrapper implementation
   */
  protected abstract Recommender buildRecommender() throws IOException, TasteException;

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany)
      throws TasteException {
    return delegate.recommend(userID, howMany);
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer)
      throws TasteException {
    return delegate.recommend(userID, howMany, rescorer);
  }

  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    return delegate.estimatePreference(userID, itemID);
  }

  @Override
  public void setPreference(long userID, long itemID, float value)
      throws TasteException {
    delegate.setPreference(userID, itemID, value);
  }

  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    delegate.removePreference(userID, itemID);
  }

  @Override
  public DataModel getDataModel() {
    return delegate.getDataModel();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    delegate.refresh(alreadyRefreshed);
  }

  protected static final File readResourceToTempFile(String resourceName) throws IOException {
    InputStream is = RecommenderWrapper.class.getResourceAsStream(resourceName);
    try {
      File tempFile = File.createTempFile("taste", null);
      tempFile.deleteOnExit();
      OutputStream os = new FileOutputStream(tempFile);
      try {
        int bytesRead;
        byte[] buffer = new byte[32768];
        while ((bytesRead = is.read(buffer)) > 0) {
          os.write(buffer, 0, bytesRead);
        }
        os.flush();
        return tempFile;
      } finally {
        IOUtils.quietClose(os);
      }
    } finally {
      IOUtils.quietClose(is);
    }
  }

}
