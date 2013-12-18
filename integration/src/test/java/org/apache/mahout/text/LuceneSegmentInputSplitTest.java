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
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.store.IOContext;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.text.doc.SingleFieldDocument;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.List;

import static java.util.Arrays.asList;

public class LuceneSegmentInputSplitTest extends AbstractLuceneStorageTest {

  private Configuration configuration;

  @Before
  public void before() throws IOException {
    configuration = getConfiguration();
  }

  @After
  public void after() throws IOException {
    HadoopUtil.delete(configuration, indexPath1);
  }

  @Test
  public void testGetSegment() throws Exception {
    SingleFieldDocument doc1 = new SingleFieldDocument("1", "This is simple document 1");
    SingleFieldDocument doc2 = new SingleFieldDocument("2", "This is simple document 2");
    SingleFieldDocument doc3 = new SingleFieldDocument("3", "This is simple document 3");

    List<SingleFieldDocument> docs = asList(doc1, doc2, doc3);
    for (SingleFieldDocument doc : docs) {
      commitDocuments(getDirectory(getIndexPath1AsFile()), doc);
    }

    assertSegmentContainsOneDoc("_0");
    assertSegmentContainsOneDoc("_1");
    assertSegmentContainsOneDoc("_2");
  }

  @Test(expected = IllegalArgumentException.class)
  public void testGetSegmentNonExistingSegment() throws Exception {
    SingleFieldDocument doc1 = new SingleFieldDocument("1", "This is simple document 1");
    SingleFieldDocument doc2 = new SingleFieldDocument("2", "This is simple document 2");
    SingleFieldDocument doc3 = new SingleFieldDocument("3", "This is simple document 3");

    List<SingleFieldDocument> docs = asList(doc1, doc2, doc3);
    for (SingleFieldDocument doc : docs) {
      commitDocuments(getDirectory(getIndexPath1AsFile()), doc);
    }

    LuceneSegmentInputSplit inputSplit = new LuceneSegmentInputSplit(indexPath1, "_3", 1000);
    inputSplit.getSegment(configuration);
  }

  private void assertSegmentContainsOneDoc(String segmentName) throws IOException {
    LuceneSegmentInputSplit inputSplit = new LuceneSegmentInputSplit(indexPath1, segmentName, 1000);
    SegmentCommitInfo segment = inputSplit.getSegment(configuration);
    SegmentReader segmentReader = new SegmentReader(segment, 1, IOContext.READ);//SegmentReader.get(true, segment, 1);
    assertEquals(segmentName, segment.info.name);
    assertEquals(1, segmentReader.numDocs());
  }


}
