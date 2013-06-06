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
import org.apache.lucene.index.SlowCompositeReaderWrapper;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.Weight;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.apache.commons.lang.StringUtils.isBlank;
import static org.apache.commons.lang.StringUtils.isNotBlank;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Generates a sequence file from a Lucene index with a specified id field as the key and a content field as the value.
 * Configure this class with a {@link LuceneStorageConfiguration} bean.
 */
public class SequenceFilesFromLuceneStorage {

  public static final String SEPARATOR_FIELDS = " ";

  private static final Logger log = LoggerFactory.getLogger(SequenceFilesFromLuceneStorage.class);

  /**
   * Generates a sequence files from a Lucene index via the given {@link LuceneStorageConfiguration}
   *
   * @param lucene2seqConf configuration bean
   * @throws java.io.IOException if index cannot be opened or sequence file could not be written
   */
  public void run(LuceneStorageConfiguration lucene2seqConf) throws IOException {
    List<Path> indexPaths = lucene2seqConf.getIndexPaths();

    for (Path indexPath : indexPaths) {
      Directory directory = FSDirectory.open(new File(indexPath.toString()));
      IndexReader reader = DirectoryReader.open(directory);
      IndexSearcher searcher = new IndexSearcher(reader);
      Configuration configuration = lucene2seqConf.getConfiguration();
      FileSystem fileSystem = FileSystem.get(configuration);
      Path sequenceFilePath = new Path(lucene2seqConf.getSequenceFilesOutputPath(), indexPath);
      SequenceFile.Writer sequenceFileWriter = new SequenceFile.Writer(fileSystem, configuration, sequenceFilePath, Text.class, Text.class);

      Text key = new Text();
      Text value = new Text();

      Weight weight = lucene2seqConf.getQuery().createWeight(searcher);
      //TODO: as the name implies, this is slow, but this is sequential anyway, so not a big deal.  Better perf. would be by looping on the segments
      AtomicReaderContext context = SlowCompositeReaderWrapper.wrap(reader).getContext();

      Scorer scorer = weight.scorer(context, true, false, null);

      if (scorer != null) {
        int processedDocs = 0;
        int docId;

        while ((docId = scorer.nextDoc()) != NO_MORE_DOCS && processedDocs < lucene2seqConf.getMaxHits()) {
          DocumentStoredFieldVisitor storedFieldVisitor = lucene2seqConf.getStoredFieldVisitor();
          reader.document(docId, storedFieldVisitor);
          Document doc = storedFieldVisitor.getDocument();
          String idValue = doc.get(lucene2seqConf.getIdField());

          StringBuilder fieldValueBuilder = new StringBuilder();
          List<String> fields = lucene2seqConf.getFields();
          for (int i = 0; i < fields.size(); i++) {
            String field = fields.get(i);
            String fieldValue = doc.get(field);
            if (isNotBlank(fieldValue)) {
              fieldValueBuilder.append(fieldValue);
              if (i != fields.size() - 1) {
                fieldValueBuilder.append(SEPARATOR_FIELDS);
              }
            }
          }

          if (isBlank(idValue) || isBlank(fieldValueBuilder.toString())) {
            continue;
          }

          key.set(idValue);
          value.set(fieldValueBuilder.toString());

          sequenceFileWriter.append(key, value);

          processedDocs++;
        }

        log.info("Wrote " + processedDocs + " documents in " + sequenceFilePath.toUri());
      } else {
        Closeables.close(sequenceFileWriter, true);
        directory.close();
        //searcher.close();
        reader.close();
        throw new RuntimeException("Could not write sequence files. Could not create scorer");
      }

      Closeables.close(sequenceFileWriter, true);
      directory.close();
      //searcher.close();
      reader.close();
    }
  }
}
