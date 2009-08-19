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

package org.apache.mahout.cf.taste.example.netflix;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FileLineIterable;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

/**
 * <p>This is a {@link DataModel} that reads the Netflix data set, as represented in its
 * unpacked file structure. Note that unpacking the distribution entails untarring the main
 * archive, then untarring the archive of training set data files.</p>
 */
public final class NetflixDataModel implements DataModel {

  private static final Logger log = LoggerFactory.getLogger(NetflixDataModel.class);

	private final DataModel delegate;
  private final boolean useSubset;

  /**
   * @param dataDirectory root directory of Netflix data set, unpacked
   * @param useSubset if true, will use only a small fraction of the data -- may be useful
   *  for quick experiments
   * @throws IOException
   */
	public NetflixDataModel(File dataDirectory, boolean useSubset) throws IOException {
		if (dataDirectory == null) {
			throw new IllegalArgumentException("dataDirectory is null");
		}
		if (!dataDirectory.exists() || !dataDirectory.isDirectory()) {
			throw new FileNotFoundException(dataDirectory.toString());
		}

    this.useSubset = useSubset;

		log.info("Creating NetflixDataModel for directory: {}", dataDirectory);

		log.info("Reading preference data...");
		FastByIDMap<PreferenceArray> users = readUsers(dataDirectory);

		log.info("Creating delegate DataModel...");
		delegate = new GenericDataModel(users);
	}

	private FastByIDMap<PreferenceArray> readUsers(File dataDirectory) {
		FastByIDMap<Collection<Preference>> userIDPrefMap = new FastByIDMap<Collection<Preference>>();

		int counter = 0;
		FilenameFilter filenameFilter = new MovieFilenameFilter();
		for (File movieFile : new File(dataDirectory, "training_set").listFiles(filenameFilter)) {
      Iterator<String> lineIterator = new FileLineIterable(movieFile, false).iterator();
			String line = lineIterator.next();
			long movieID = Long.parseLong(line.substring(0, line.length() - 1)); // strip colon
			while (lineIterator.hasNext()) {
        line = lineIterator.next();
				if (++counter % 100000 == 0) {
					log.info("Processed {} prefs", counter);
				}
				int firstComma = line.indexOf((int) ',');
				long userID = Long.parseLong(line.substring(0, firstComma));
				int secondComma = line.indexOf((int) ',', firstComma + 1);
				float rating = Float.parseFloat(line.substring(firstComma + 1, secondComma));
				Collection<Preference> userPrefs = userIDPrefMap.get(userID);
				if (userPrefs == null) {
					userPrefs = new ArrayList<Preference>(2);
					userIDPrefMap.put(userID, userPrefs);
				}
				userPrefs.add(new GenericPreference(userID, movieID, rating));
			}
		}

		return GenericDataModel.toDataMap(userIDPrefMap, true);
	}

	@Override
  public LongPrimitiveIterator getUserIDs() throws TasteException {
		return delegate.getUserIDs();
	}

	@Override
  public PreferenceArray getPreferencesFromUser(long id) throws TasteException {
		return delegate.getPreferencesFromUser(id);
	}

  @Override
  public FastIDSet getItemIDsFromUser(long userID) throws TasteException {
    return delegate.getItemIDsFromUser(userID);
  }

  @Override
  public LongPrimitiveIterator getItemIDs() throws TasteException {
		return delegate.getItemIDs();
	}

	@Override
  public PreferenceArray getPreferencesForItem(long itemID) throws TasteException {
		return delegate.getPreferencesForItem(itemID);
	}

  @Override
  public Float getPreferenceValue(long userID, long itemID) throws TasteException {
    return delegate.getPreferenceValue(userID, itemID);
  }

  @Override
  public int getNumItems() throws TasteException {
		return delegate.getNumItems();
	}

	@Override
  public int getNumUsers() throws TasteException {
		return delegate.getNumUsers();
	}

  @Override
  public int getNumUsersWithPreferenceFor(long... itemIDs) throws TasteException {
    return delegate.getNumUsersWithPreferenceFor(itemIDs);
  }

  @Override
  public void setPreference(long userID, long itemID, float value) {
		throw new UnsupportedOperationException();
	}

	@Override
  public void removePreference(long userID, long itemID) {
		throw new UnsupportedOperationException();
	}

	@Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
		// do nothing
	}

  private class MovieFilenameFilter implements FilenameFilter {
    @Override
      public boolean accept(File dir, String filename) {
      return filename.startsWith(useSubset ? "mv_0000" : "mv_");
    }
  }

  @Override
  public String toString() {
    return "NetflixDataModel";
  }

}
