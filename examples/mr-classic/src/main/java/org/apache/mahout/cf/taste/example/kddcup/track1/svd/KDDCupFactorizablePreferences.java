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

package org.apache.mahout.cf.taste.example.kddcup.track1.svd;

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import org.apache.mahout.cf.taste.example.kddcup.DataFileIterable;
import org.apache.mahout.cf.taste.impl.common.AbstractLongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.Pair;

import java.io.File;

public class KDDCupFactorizablePreferences implements FactorizablePreferences {

  private final File dataFile;

  public KDDCupFactorizablePreferences(File dataFile) {
    this.dataFile = dataFile;
  }

  @Override
  public LongPrimitiveIterator getUserIDs() {
    return new FixedSizeLongIterator(numUsers());
  }

  @Override
  public LongPrimitiveIterator getItemIDs() {
    return new FixedSizeLongIterator(numItems());
  }

  @Override
  public Iterable<Preference> getPreferences() {
    Iterable<Iterable<Preference>> prefIterators =
        Iterables.transform(new DataFileIterable(dataFile),
          new Function<Pair<PreferenceArray,long[]>,Iterable<Preference>>() {
            @Override
            public Iterable<Preference> apply(Pair<PreferenceArray,long[]> from) {
              return from.getFirst();
            }
          });
    return Iterables.concat(prefIterators);
  }

  @Override
  public float getMinPreference() {
    return 0;
  }

  @Override
  public float getMaxPreference() {
    return 100;
  }

  @Override
  public int numUsers() {
    return 1000990;
  }

  @Override
  public int numItems() {
    return 624961;
  }

  @Override
  public int numPreferences() {
    return 252800275;
  }

  static class FixedSizeLongIterator extends AbstractLongPrimitiveIterator {

    private long currentValue;
    private final long maximum;

    FixedSizeLongIterator(long maximum) {
      this.maximum = maximum;
      currentValue = 0;
    }

    @Override
    public long nextLong() {
      return currentValue++;
    }

    @Override
    public long peek() {
      return currentValue;
    }

    @Override
    public void skip(int n) {
      currentValue += n;
    }

    @Override
    public boolean hasNext() {
      return currentValue < maximum;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

}
