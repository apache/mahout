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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FileLineIterable;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Collection;
import java.util.Iterator;
import java.util.logging.Logger;

/**
 * <p>This is a {@link DataModel} that reads the Netflix data set, as represented in its
 * unpacked file structure. Note that unpacking the distribution entails untarring the main
 * archive, then untarring the archive of training set data files.</p>
 */
public final class NetflixDataModel implements DataModel {

	private static final Logger log = Logger.getLogger(NetflixDataModel.class.getName());

	private final DataModel delegate;
  private final boolean useSubset;

  /**
   * @param dataDirectory root directory of Netflix data set, unpacked
   * @param useSubset if true, will use only a small fraction of the data -- may be useful
   *  for quick experiments
   */
	public NetflixDataModel(File dataDirectory, boolean useSubset) throws IOException {
		if (dataDirectory == null) {
			throw new IllegalArgumentException("dataDirectory is null");
		}
		if (!dataDirectory.exists() || !dataDirectory.isDirectory()) {
			throw new FileNotFoundException(dataDirectory.toString());
		}

    this.useSubset = useSubset;

		log.info("Creating NetflixDataModel for directory: " + dataDirectory);

		log.info("Reading movie data...");
		List<NetflixMovie> movies = readMovies(dataDirectory);

		log.info("Reading preference data...");
		List<User> users = readUsers(dataDirectory, movies);

		log.info("Creating delegate DataModel...");
		delegate = new GenericDataModel(users);
	}

	private List<User> readUsers(File dataDirectory, List<NetflixMovie> movies) throws IOException {
		Map<Integer, List<Preference>> userIDPrefMap = new FastMap<Integer, List<Preference>>();

		int counter = 0;
		FilenameFilter filenameFilter = new MovieFilenameFilter();
		for (File movieFile : new File(dataDirectory, "training_set").listFiles(filenameFilter)) {
      Iterator<String> lineIterator = new FileLineIterable(movieFile, false).iterator();
			String line = lineIterator.next();
			if (line == null) {
				throw new IOException("Can't read first line of file " + movieFile);
			}
			int movieID = Integer.parseInt(line.substring(0, line.length() - 1));
			NetflixMovie movie = movies.get(movieID - 1);
			if (movie == null) {
				throw new IllegalArgumentException("No such movie: " + movieID);
			}
			while (lineIterator.hasNext()) {
        line = lineIterator.next();
				counter++;
				if (counter % 100000 == 0) {
					log.info("Processed " + counter + " prefs");
				}
				int firstComma = line.indexOf((int) ',');
				Integer userID = Integer.valueOf(line.substring(0, firstComma));
				int secondComma = line.indexOf((int) ',', firstComma + 1);
				double rating = Double.parseDouble(line.substring(firstComma + 1, secondComma));
				List<Preference> userPrefs = userIDPrefMap.get(userID);
				if (userPrefs == null) {
					userPrefs = new ArrayList<Preference>();
					userIDPrefMap.put(userID, userPrefs);
				}
				userPrefs.add(new GenericPreference(null, movie, rating));
			}
		}

		List<User> users = new ArrayList<User>(userIDPrefMap.size());
		for (Map.Entry<Integer, List<Preference>> entry : userIDPrefMap.entrySet()) {
			users.add(new GenericUser<Integer>(entry.getKey(), entry.getValue()));
		}
		return users;
	}

	private static List<NetflixMovie> readMovies(File dataDirectory) {
		List<NetflixMovie> movies = new ArrayList<NetflixMovie>();
    for (String line : new FileLineIterable(new File(dataDirectory, "movie_titles.txt"), false)) {
			int firstComma = line.indexOf((int) ',');
			Integer id = Integer.valueOf(line.substring(0, firstComma));
			int secondComma = line.indexOf((int) ',', firstComma + 1);
			String title = line.substring(secondComma + 1);
			movies.add(new NetflixMovie(id, title));
		}
		return movies;
	}

	@Override
  public Iterable<? extends User> getUsers() throws TasteException {
		return delegate.getUsers();
	}

	@Override
  public User getUser(Object id) throws TasteException {
		return delegate.getUser(id);
	}

	@Override
  public Iterable<? extends Item> getItems() throws TasteException {
		return delegate.getItems();
	}

	@Override
  public Item getItem(Object id) throws TasteException {
		return delegate.getItem(id);
	}

	@Override
  public Iterable<? extends Preference> getPreferencesForItem(Object itemID) throws TasteException {
		return delegate.getPreferencesForItem(itemID);
	}

	@Override
  public Preference[] getPreferencesForItemAsArray(Object itemID) throws TasteException {
		return delegate.getPreferencesForItemAsArray(itemID);
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
  public int getNumUsersWithPreferenceFor(Object... itemIDs) throws TasteException {
    return delegate.getNumUsersWithPreferenceFor(itemIDs);
  }

  @Override
  public void setPreference(Object userID, Object itemID, double value) {
		throw new UnsupportedOperationException();
	}

	@Override
  public void removePreference(Object userID, Object itemID) {
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
}
