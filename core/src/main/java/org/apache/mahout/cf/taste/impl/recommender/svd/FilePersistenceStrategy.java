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

package org.apache.mahout.cf.taste.impl.recommender.svd;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Map;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Provides a file-based persistent store. */
public class FilePersistenceStrategy implements PersistenceStrategy {

  private final File file;

  private static final Logger log = LoggerFactory.getLogger(FilePersistenceStrategy.class);

  /**
   * @param file the file to use for storage. If the file does not exist it will be created when required.
   */
  public FilePersistenceStrategy(File file) {
    this.file = Preconditions.checkNotNull(file);
  }

  @Override
  public Factorization load() throws IOException {
    if (!file.exists()) {
      log.info("{} does not yet exist, no factorization found", file.getAbsolutePath());
      return null;
    }
    DataInputStream in = null;
    try {
      log.info("Reading factorization from {}...", file.getAbsolutePath());
      in = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
      return readBinary(in);
    } finally {
      Closeables.close(in, true);
    }
  }

  @Override
  public void maybePersist(Factorization factorization) throws IOException {
    DataOutputStream out = null;
    try {
      log.info("Writing factorization to {}...", file.getAbsolutePath());
      out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
      writeBinary(factorization, out);
    } finally {
      Closeables.close(out, false);
    }
  }

  protected static void writeBinary(Factorization factorization, DataOutput out) throws IOException {
    out.writeInt(factorization.numFeatures());
    out.writeInt(factorization.numUsers());
    out.writeInt(factorization.numItems());

    for (Map.Entry<Long,Integer> mappingEntry : factorization.getUserIDMappings()) {
      long userID = mappingEntry.getKey();
      out.writeInt(mappingEntry.getValue());
      out.writeLong(userID);
      try {
        double[] userFeatures = factorization.getUserFeatures(userID);
        for (int feature = 0; feature < factorization.numFeatures(); feature++) {
          out.writeDouble(userFeatures[feature]);
        }
      } catch (NoSuchUserException e) {
        throw new IOException("Unable to persist factorization", e);
      }
    }

    for (Map.Entry<Long,Integer> entry : factorization.getItemIDMappings()) {
      long itemID = entry.getKey();
      out.writeInt(entry.getValue());
      out.writeLong(itemID);
      try {
        double[] itemFeatures = factorization.getItemFeatures(itemID);
        for (int feature = 0; feature < factorization.numFeatures(); feature++) {
          out.writeDouble(itemFeatures[feature]);
        }
      } catch (NoSuchItemException e) {
        throw new IOException("Unable to persist factorization", e);
      }
    }
  }

  public static Factorization readBinary(DataInput in) throws IOException {
    int numFeatures = in.readInt();
    int numUsers = in.readInt();
    int numItems = in.readInt();

    FastByIDMap<Integer> userIDMapping = new FastByIDMap<Integer>(numUsers);
    double[][] userFeatures = new double[numUsers][numFeatures];

    for (int n = 0; n < numUsers; n++) {
      int userIndex = in.readInt();
      long userID = in.readLong();
      userIDMapping.put(userID, userIndex);
      for (int feature = 0; feature < numFeatures; feature++) {
        userFeatures[userIndex][feature] = in.readDouble();
      }
    }

    FastByIDMap<Integer> itemIDMapping = new FastByIDMap<Integer>(numItems);
    double[][] itemFeatures = new double[numItems][numFeatures];

    for (int n = 0; n < numItems; n++) {
      int itemIndex = in.readInt();
      long itemID = in.readLong();
      itemIDMapping.put(itemID, itemIndex);
      for (int feature = 0; feature < numFeatures; feature++) {
        itemFeatures[itemIndex][feature] = in.readDouble();
      }
    }

    return new Factorization(userIDMapping, itemIDMapping, userFeatures, itemFeatures);
  }

}
