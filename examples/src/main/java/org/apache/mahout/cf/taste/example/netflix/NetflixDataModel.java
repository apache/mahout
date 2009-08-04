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
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FastSet;
import org.apache.mahout.cf.taste.impl.common.FileLineIterable;
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
import java.util.List;
import java.util.Map;

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
		Map<Comparable<?>, PreferenceArray> users = readUsers(dataDirectory);

		log.info("Creating delegate DataModel...");
		delegate = new GenericDataModel(users);
	}

  static List<Comparable<?>> readMovies(File dataDirectory) {
		List<Comparable<?>> movies = new ArrayList<Comparable<?>>(17770);
    for (String line : new FileLineIterable(new File(dataDirectory, "movie_titles.txt"), false)) {
			int firstComma = line.indexOf((int) ',');
			Integer id = Integer.valueOf(line.substring(0, firstComma));
			movies.add(id);
      if (id != movies.size()) {
        throw new IllegalStateException("A movie is missing from movie_titles.txt");
      }
		}
		return movies;
	}

	private Map<Comparable<?>, PreferenceArray> readUsers(File dataDirectory) throws IOException {
		Map<Comparable<?>, Collection<Preference>> userIDPrefMap = new FastMap<Comparable<?>, Collection<Preference>>();

		int counter = 0;
		FilenameFilter filenameFilter = new MovieFilenameFilter();
		for (File movieFile : new File(dataDirectory, "training_set").listFiles(filenameFilter)) {
      Iterator<String> lineIterator = new FileLineIterable(movieFile, false).iterator();
			String line = lineIterator.next();
			Integer movieID = Integer.valueOf(line.substring(0, line.length() - 1)); // strip colon
			while (lineIterator.hasNext()) {
        line = lineIterator.next();
				counter++;
				if (counter % 100000 == 0) {
					log.info("Processed {} prefs", counter);
				}
				int firstComma = line.indexOf((int) ',');
				Integer userID = Integer.valueOf(line.substring(0, firstComma));
				int secondComma = line.indexOf((int) ',', firstComma + 1);
				float rating = Float.parseFloat(line.substring(firstComma + 1, secondComma));
				Collection<Preference> userPrefs = userIDPrefMap.get(userID);
				if (userPrefs == null) {
					userPrefs = new ArrayList<Preference>();
					userIDPrefMap.put(userID, userPrefs);
				}
				userPrefs.add(new GenericPreference(null, movieID, rating));
			}
		}

		return GenericDataModel.toPrefArrayValues(userIDPrefMap, true);
	}

	@Override
  public Iterable<Comparable<?>> getUserIDs() throws TasteException {
		return delegate.getUserIDs();
	}

	@Override
  public PreferenceArray getPreferencesFromUser(Comparable<?> id) throws TasteException {
		return delegate.getPreferencesFromUser(id);
	}

  @Override
  public FastSet<Comparable<?>> getItemIDsFromUser(Comparable<?> userID) throws TasteException {
    return delegate.getItemIDsFromUser(userID);
  }

  @Override
  public Iterable<Comparable<?>> getItemIDs() throws TasteException {
		return delegate.getItemIDs();
	}

	@Override
  public PreferenceArray getPreferencesForItem(Comparable<?> itemID) throws TasteException {
		return delegate.getPreferencesForItem(itemID);
	}

  @Override
  public Float getPreferenceValue(Comparable<?> userID, Comparable<?> itemID) throws TasteException {
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
  public int getNumUsersWithPreferenceFor(Comparable<?>... itemIDs) throws TasteException {
    return delegate.getNumUsersWithPreferenceFor(itemIDs);
  }

  @Override
  public void setPreference(Comparable<?> userID, Comparable<?> itemID, float value) {
		throw new UnsupportedOperationException();
	}

	@Override
  public void removePreference(Comparable<?> userID, Comparable<?> itemID) {
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
