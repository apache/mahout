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

import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.impl.common.FileLineIterable;

import java.util.List;
import java.util.ArrayList;
import java.io.File;

final class NetflixMovie implements Item {

	private final Integer id;
	private final String title;

	NetflixMovie(Integer id, String title) {
		if (id == null || title == null) {
			throw new IllegalArgumentException("ID or title is null");
		}
		this.id = id;
		this.title = title;
	}

	@Override
  public Object getID() {
		return id;
	}

	String getTitle() {
		return title;
	}

	@Override
  public boolean isRecommendable() {
		return true;
	}

	@Override
	public int hashCode() {
		return id.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		return obj instanceof NetflixMovie && ((NetflixMovie) obj).id.equals(id);
	}

	@Override
  public int compareTo(Item item) {
		return this.id.compareTo(((NetflixMovie) item).id);
	}

	@Override
	public String toString() {
		return id + ":" + title;
	}

  static List<NetflixMovie> readMovies(File dataDirectory) {
		List<NetflixMovie> movies = new ArrayList<NetflixMovie>(17770);
    for (String line : new FileLineIterable(new File(dataDirectory, "movie_titles.txt"), false)) {
			int firstComma = line.indexOf((int) ',');
			int id = Integer.parseInt(line.substring(0, firstComma));
			int secondComma = line.indexOf((int) ',', firstComma + 1);
			String title = line.substring(secondComma + 1);
			movies.add(new NetflixMovie(id, title));
      if (id != movies.size()) {
        throw new IllegalStateException("A movie is missing from movie_titles.txt");
      }
		}
		return movies;
	}

}
