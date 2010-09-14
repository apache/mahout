/*
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

/*
 * ============================================================================
 * Derived from DevRandomSeedGenerator which has this copyright notice
 *
 *    Copyright 2006-2009 Daniel W. Dyer
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 * ============================================================================
 */

package org.apache.mahout.common;

import org.uncommons.maths.random.SeedException;
import org.uncommons.maths.random.SeedGenerator;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * RNG seed strategy that gets data from {@literal /dev/urandom} on systems that provide it (e.g.
 * Solaris/Linux).  If {@literal /dev/random} does not exist or is not accessible, a {@link
 * SeedException} is thrown.  The point of pulling from /dev/urandom instead of from /dev/random
 * is that /dev/random will block if it doesn't think it has enough entropy.  In most production
 * applications of Mahout, that really isn't necessary.
 */
public final class DevURandomSeedGenerator implements SeedGenerator {

  private static final File DEV_URANDOM = new File("/dev/urandom");

  /**
   * @return The requested number of random bytes, read directly from {@literal /dev/urandom}.
   * @throws SeedException If {@literal /dev/urandom} does not exist or is not accessible
   */
  public byte[] generateSeed(int length) throws SeedException {
    FileInputStream file = null;
    try {
      file = new FileInputStream(DEV_URANDOM);
      byte[] randomSeed = new byte[length];
      int count = 0;
      while (count < length) {
        int bytesRead = file.read(randomSeed, count, length - count);
        if (bytesRead == -1) {
          throw new SeedException("EOF encountered reading random data.");
        }
        count += bytesRead;
      }
      return randomSeed;
    } catch (IOException ex) {
      throw new SeedException("Failed reading from " + DEV_URANDOM.getName(), ex);
    } catch (SecurityException ex) {
      // Might be thrown if resource access is restricted (such as in
      // an applet sandbox).
      throw new SeedException("SecurityManager prevented access to " + DEV_URANDOM.getName(), ex);
    } finally {
      if (file != null) {
        try {
          file.close();
        } catch (IOException ex) {
          // Ignore.
        }
      }
    }
  }

  @Override
  public String toString() {
    return "/dev/urandom";
  }
}
