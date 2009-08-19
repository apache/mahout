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

package org.apache.mahout.cf.taste.impl.model;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.IDMigrator;

import java.nio.charset.Charset;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public abstract class AbstractIDMigrator implements IDMigrator {

  private static final Charset UTF8_CHARSET = Charset.forName("UTF8");

  private final MessageDigest md5Digest;

  protected AbstractIDMigrator() {
    try {
      md5Digest = MessageDigest.getInstance("MD5");
    } catch (NoSuchAlgorithmException nsae) {
      // Can't happen
      throw new IllegalStateException(nsae);
    }
  }

  protected final long hash(String value) {
    byte[] md5hash;
    synchronized (md5Digest) {
      md5hash = md5Digest.digest(value.getBytes(UTF8_CHARSET));
      md5Digest.reset();
    }
    long hash = 0L;
    for (int i = 0; i < 8; i++) {
      hash = (hash << 8) | (md5hash[i] & 0x00000000000000FFL);
    }
    return hash;
  }

  @Override
  public long toLongID(String stringID) throws TasteException {
    long longID = hash(stringID);
    storeMapping(longID, stringID);
    return longID;
  }

  @Override
  public void initialize(Iterable<String> stringIDs) throws TasteException {
    for (String stringID : stringIDs) {
      toLongID(stringID);
    }
  }

  protected abstract void storeMapping(long longID, String stringID) throws TasteException;

}