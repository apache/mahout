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

import com.google.common.io.Files;
import com.google.common.io.InputSupplier;
import com.google.common.io.Resources;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
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

  private static final Logger log = LoggerFactory.getLogger(RecommenderWrapper.class);

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
  public List<RecommendedItem> recommend(long userID, int howMany) throws TasteException {
    return delegate.recommend(userID, howMany);
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) throws TasteException {
    return delegate.recommend(userID, howMany, rescorer);
  }

  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    return delegate.estimatePreference(userID, itemID);
  }

  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
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

  /**
   * Reads the given resource into a temporary file. This is intended to be used
   * to read data files which are stored as a resource available on the classpath,
   * such as in a JAR file. However for convenience the resource name will also
   * be interpreted as a relative path to a local file, if no such resource is
   * found. This facilitates testing.
   *
   * @param resourceName name of resource in classpath, or relative path to file
   * @return temporary {@link File} with resource data
   * @throws IOException if an error occurs while reading or writing data
   */
  public static File readResourceToTempFile(String resourceName) throws IOException {
    String absoluteResource = resourceName.startsWith("/") ? resourceName : '/' + resourceName;
    log.info("Loading resource {}", absoluteResource);
    InputSupplier<? extends InputStream> inSupplier;
    try {
      URL resourceURL = Resources.getResource(RecommenderWrapper.class, absoluteResource);
      inSupplier = Resources.newInputStreamSupplier(resourceURL);
    } catch (IllegalArgumentException iae) {
      File resourceFile = new File(resourceName);
      log.info("Falling back to load file {}", resourceFile.getAbsolutePath());
      inSupplier = Files.newInputStreamSupplier(resourceFile);
    }
    File tempFile = File.createTempFile("taste", null);
    tempFile.deleteOnExit();
    Files.copy(inSupplier, tempFile);
    return tempFile;
  }

}
