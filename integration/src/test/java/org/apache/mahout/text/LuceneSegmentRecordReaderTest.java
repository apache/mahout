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
package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.TaskAttemptID;
import org.apache.lucene.index.*;
import org.apache.mahout.common.HadoopUtil;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;

import static java.util.Arrays.asList;

public class LuceneSegmentRecordReaderTest extends AbstractLuceneStorageTest {
  private Configuration configuration;


  @Before
  public void before() throws IOException, InterruptedException {
    LuceneStorageConfiguration lucene2SeqConf = new
    LuceneStorageConfiguration(getConfiguration(), asList(getIndexPath1()), new Path("output"), "id", asList("field"));
    configuration = lucene2SeqConf.serialize();
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs.subList(0, 500));
    commitDocuments(getDirectory(getIndexPath1AsFile()), docs.subList(500, 1000));

  }

  @After
  public void after() throws IOException {
    HadoopUtil.delete(configuration, getIndexPath1());
  }

  @Test
  public void testKey() throws Exception {
    LuceneSegmentRecordReader recordReader = new LuceneSegmentRecordReader();
    SegmentInfos segmentInfos = new SegmentInfos();
    segmentInfos.read(getDirectory(getIndexPath1AsFile()));
    for (SegmentInfoPerCommit segmentInfo : segmentInfos) {
      int docId = 0;
      LuceneSegmentInputSplit inputSplit = new LuceneSegmentInputSplit(getIndexPath1(), segmentInfo.info.name, segmentInfo.sizeInBytes());
      TaskAttemptContext context = getTaskAttemptContext(configuration, new TaskAttemptID());
      recordReader.initialize(inputSplit, context);
      for (int i = 0; i < 500; i++){
        recordReader.nextKeyValue();
        //we can't be sure of the order we are getting the segments, so we have to fudge here a bit on the id, but it is either id: i or i + 500
        assertTrue("i = " + i + " docId= " + docId, String.valueOf(docId).equals(recordReader.getCurrentKey().toString()) || String.valueOf(docId+500).equals(recordReader.getCurrentKey().toString()));
        assertEquals(NullWritable.get(), recordReader.getCurrentValue());
        docId++;
      }
    }
  }

  // Use reflection to abstract this incompatibility between Hadoop 1 & 2 APIs.
  private TaskAttemptContext getTaskAttemptContext(Configuration conf, TaskAttemptID jobID) throws
      ClassNotFoundException, NoSuchMethodException, IllegalAccessException,
      InvocationTargetException, InstantiationException {
    Class<? extends TaskAttemptContext> clazz = null;
    if (!TaskAttemptContext.class.isInterface()) {
      clazz = TaskAttemptContext.class;
    } else {
      clazz = (Class<? extends TaskAttemptContext>)
          Class.forName("org.apache.hadoop.mapreduce.task.TaskAttemptContextImpl");
    }
    return clazz.getConstructor(Configuration.class, TaskAttemptID.class)
        .newInstance(conf, jobID);
  }
}
