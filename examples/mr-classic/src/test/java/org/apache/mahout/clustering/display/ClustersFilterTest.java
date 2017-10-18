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

package org.apache.mahout.clustering.display;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

public class ClustersFilterTest extends MahoutTestCase {

  private Configuration configuration;
  private Path output;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    configuration = getConfiguration();
    output = getTestTempDirPath();
  }

  @Test
  public void testAcceptNotFinal() throws Exception {
    Path path0 = new Path(output, "clusters-0");
    Path path1 = new Path(output, "clusters-1");

    path0.getFileSystem(configuration).createNewFile(path0);
    path1.getFileSystem(configuration).createNewFile(path1);

    PathFilter clustersFilter = new ClustersFilter();

    assertTrue(clustersFilter.accept(path0));
    assertTrue(clustersFilter.accept(path1));
  }

  @Test
  public void testAcceptFinalPath() throws IOException {
    Path path0 = new Path(output, "clusters-0");
    Path path1 = new Path(output, "clusters-1");
    Path path2 = new Path(output, "clusters-2");
    Path path3Final = new Path(output, "clusters-3-final");

    path0.getFileSystem(configuration).createNewFile(path0);
    path1.getFileSystem(configuration).createNewFile(path1);
    path2.getFileSystem(configuration).createNewFile(path2);
    path3Final.getFileSystem(configuration).createNewFile(path3Final);

    PathFilter clustersFilter = new ClustersFilter();

    assertTrue(clustersFilter.accept(path0));
    assertTrue(clustersFilter.accept(path1));
    assertTrue(clustersFilter.accept(path2));
    assertTrue(clustersFilter.accept(path3Final));
  }
}
