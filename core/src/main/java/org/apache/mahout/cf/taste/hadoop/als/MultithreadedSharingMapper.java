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

package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.util.ReflectionUtils;

import java.io.IOException;

/**
 * Multithreaded Mapper for {@link SharingMapper}s. Will call setupSharedInstance() once in the controlling thread
 * before executing the mappers using a thread pool.
 *
 * @param <K1>
 * @param <V1>
 * @param <K2>
 * @param <V2>
 */
public class MultithreadedSharingMapper<K1, V1, K2, V2> extends MultithreadedMapper<K1, V1, K2, V2> {

  private static final String MAPPER_CLASS = "mapred.map.multithreadedrunner.class";

  @Override
  public void run(Context ctx) throws IOException, InterruptedException {

    Configuration conf = ctx.getConfiguration();

    Class<? extends SharingMapper<K1,V1,K2,V2, ?>> mapperClass =
        (Class<SharingMapper<K1,V1,K2,V2, ?>>) conf.getClass(MAPPER_CLASS, SharingMapper.class);
    // instantiate the mapper
    SharingMapper<K1,V1,K2,V2, ?> mapper = ReflectionUtils.newInstance(mapperClass, conf);

    // single threaded call to setup the sharing mapper
    mapper.setupSharedInstance(ctx);

    // multithreaded execution
    super.run(ctx);
  }
}
