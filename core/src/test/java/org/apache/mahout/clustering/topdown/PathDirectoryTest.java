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

import static java.io.File.separator;
import static org.apache.mahout.clustering.topdown.PathDirectory.getClusterPostProcessorOutputDirectory;
import static org.apache.mahout.clustering.topdown.TopDownClusteringPathConstants.clusteredPointsDirectory;
import static org.apache.mahout.clustering.topdown.TopDownClusteringPathConstants.postProcessDirectory;
import static org.apache.mahout.clustering.topdown.TopDownClusteringPathConstants.topLevelClusterDirectory;
import junit.framework.Assert;

import org.apache.hadoop.fs.Path;
import org.junit.Test;

public class PathDirectoryTest {
  
  private Path output = new Path("output");
  
  @Test
  public void shouldReturnTopLevelClusterPath() {
    Path expectedPath = new Path(output, topLevelClusterDirectory);
    Assert.assertEquals(expectedPath, PathDirectory.getTopLevelClusterPath(output));
  }
  
  @Test
  public void shouldReturnClusterPostProcessorOutputDirectory() {
    Path expectedPath = new Path(output, postProcessDirectory);
    Assert.assertEquals(expectedPath, getClusterPostProcessorOutputDirectory(output));
  }
  
  @Test
  public void shouldReturnClusterOutputClusteredPoints() {
    Path expectedPath = new Path(output, clusteredPointsDirectory + separator + "*");
    Assert.assertEquals(expectedPath, PathDirectory.getClusterOutputClusteredPoints(output));
  }
  
  @Test
  public void shouldReturnBottomLevelClusterPath() {
    Path expectedPath = new Path(output + separator
                                 + TopDownClusteringPathConstants.bottomLevelClusterDirectory + separator
                                 + "1");
    Assert.assertEquals(expectedPath, PathDirectory.getBottomLevelClusterPath(output, "1"));
  }
  
  @Test
  public void shouldReturnClusterPathForClusterId() {
    Path expectedPath = new Path(getClusterPostProcessorOutputDirectory(output), new Path("1"));
    Assert.assertEquals(expectedPath,
      PathDirectory.getClusterPathForClusterId(getClusterPostProcessorOutputDirectory(output), "1"));
  }
  
}
