package org.apache.mahout.text;
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

import java.io.File;
import java.io.IOException;
import java.util.List;

import com.google.common.base.Strings;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.DocumentStoredFieldVisitor;
import org.apache.lucene.index.AtomicReaderContext;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.search.Collector;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.apache.commons.lang.StringUtils.isBlank;

/**
 * Generates a sequence file from a Lucene index with a specified id field as the key and a content field as the value.
 * Configure this class with a {@link LuceneStorageConfiguration} bean.
 */
public class SequenceFilesFromLuceneStorage {
  private static final Logger log = LoggerFactory.getLogger(SequenceFilesFromLuceneStorage.class);

  /**
   * Generates a sequence files from a Lucene index via the given {@link LuceneStorageConfiguration}
   *
   * @param lucene2seqConf configuration bean
   * @throws java.io.IOException if index cannot be opened or sequence file could not be written
   */
  public void run(final LuceneStorageConfiguration lucene2seqConf) throws IOException {
    List<Path> indexPaths = lucene2seqConf.getIndexPaths();
    int processedDocs = 0;

    for (Path indexPath : indexPaths) {
      Directory directory = FSDirectory.open(new File(indexPath.toUri().getPath()));
      IndexReader reader = DirectoryReader.open(directory);
      IndexSearcher searcher = new IndexSearcher(reader);

      LuceneIndexHelper.fieldShouldExistInIndex(searcher, lucene2seqConf.getIdField());
      for (String field : lucene2seqConf.getFields()) {
        LuceneIndexHelper.fieldShouldExistInIndex(searcher, field);
      }

      Configuration configuration = lucene2seqConf.getConfiguration();
      FileSystem fileSystem = FileSystem.get(configuration);
      Path sequenceFilePath = new Path(lucene2seqConf.getSequenceFilesOutputPath(), indexPath.getName());
      final SequenceFile.Writer sequenceFileWriter = new SequenceFile.Writer(fileSystem, configuration,
          sequenceFilePath, Text.class, Text.class);

      SeqFileWriterCollector writerCollector = new SeqFileWriterCollector(lucene2seqConf, sequenceFileWriter,
          processedDocs);
      searcher.search(lucene2seqConf.getQuery(), writerCollector);
      log.info("Wrote " + writerCollector.processedDocs + " documents in " + sequenceFilePath.toUri());
      processedDocs = writerCollector.processedDocs;
      Closeables.close(sequenceFileWriter, false);
      directory.close();
      //searcher.close();
      reader.close();
    }
  }

  private static class SeqFileWriterCollector extends Collector {
    private final LuceneStorageConfiguration lucene2seqConf;
    private final SequenceFile.Writer sequenceFileWriter;
    public int processedDocs;
    AtomicReaderContext arc;

    SeqFileWriterCollector(LuceneStorageConfiguration lucene2seqConf, SequenceFile.Writer sequenceFileWriter,
                           int processedDocs) {
      this.lucene2seqConf = lucene2seqConf;
      this.sequenceFileWriter = sequenceFileWriter;
      this.processedDocs = processedDocs;
    }

    @Override
    public void setScorer(Scorer scorer) throws IOException {
      //don't care about scoring, we just want the matches
    }

    @Override
    public void collect(int docNum) throws IOException {
      if (processedDocs < lucene2seqConf.getMaxHits()) {
        final DocumentStoredFieldVisitor storedFieldVisitor = lucene2seqConf.getStoredFieldVisitor();
        arc.reader().document(docNum, storedFieldVisitor);

        Document doc = storedFieldVisitor.getDocument();
        List<String> fields = lucene2seqConf.getFields();
        Text theKey = new Text(Strings.nullToEmpty(doc.get(lucene2seqConf.getIdField())));
        Text theValue = new Text();
        LuceneSeqFileHelper.populateValues(doc, theValue, fields);
        //if they are both empty, don't write
        if (isBlank(theKey.toString()) && isBlank(theValue.toString())) {
          return;
        }
        sequenceFileWriter.append(theKey, theValue);
        processedDocs++;
      }
    }

    @Override
    public void setNextReader(AtomicReaderContext context) throws IOException {
      arc = context;
    }

    @Override
    public boolean acceptsDocsOutOfOrder() {
      return true;
    }
  }
}
