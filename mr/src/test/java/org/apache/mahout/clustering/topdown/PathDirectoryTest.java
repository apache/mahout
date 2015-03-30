/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.mahout.clustering.topdown;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

import java.io.File;

public final class PathDirectoryTest extends MahoutTestCase {
  
  private final Path output = new Path("output");
  
  @Test
  public void shouldReturnTopLevelClusterPath() {
    Path expectedPath = new Path(output, PathDirectory.TOP_LEVEL_CLUSTER_DIRECTORY);
    assertEquals(expectedPath, PathDirectory.getTopLevelClusterPath(output));
  }
  
  @Test
  public void shouldReturnClusterPostProcessorOutputDirectory() {
    Path expectedPath = new Path(output, PathDirectory.POST_PROCESS_DIRECTORY);
    assertEquals(expectedPath, PathDirectory.getClusterPostProcessorOutputDirectory(output));
  }
  
  @Test
  public void shouldReturnClusterOutputClusteredPoints() {
    Path expectedPath = new Path(output, PathDirectory.CLUSTERED_POINTS_DIRECTORY + File.separator + '*');
    assertEquals(expectedPath, PathDirectory.getClusterOutputClusteredPoints(output));
  }
  
  @Test
  public void shouldReturnBottomLevelClusterPath() {
    Path expectedPath = new Path(output + File.separator
                                 + PathDirectory.BOTTOM_LEVEL_CLUSTER_DIRECTORY + File.separator
                                 + '1');
    assertEquals(expectedPath, PathDirectory.getBottomLevelClusterPath(output, "1"));
  }
  
  @Test
  public void shouldReturnClusterPathForClusterId() {
    Path expectedPath = new Path(PathDirectory.getClusterPostProcessorOutputDirectory(output), new Path("1"));
    assertEquals(expectedPath, PathDirectory.getClusterPathForClusterId(
        PathDirectory.getClusterPostProcessorOutputDirectory(output), "1"));
  }
  
}
