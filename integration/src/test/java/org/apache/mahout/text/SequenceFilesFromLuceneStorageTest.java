package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TermQuery;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.text.doc.MultipleFieldsDocument;
import org.apache.mahout.text.doc.NumericFieldDocument;
import org.apache.mahout.text.doc.SingleFieldDocument;
import org.apache.mahout.text.doc.UnstoredFieldsDocument;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Iterator;

import static java.util.Arrays.asList;
import static org.junit.Assert.assertFalse;

public class SequenceFilesFromLuceneStorageTest extends AbstractLuceneStorageTest {

  private SequenceFilesFromLuceneStorage lucene2Seq;
  private LuceneStorageConfiguration lucene2SeqConf;

  private SingleFieldDocument document1;
  private SingleFieldDocument document2;
  private SingleFieldDocument document3;
  private Path seqFilesOutputPath;
  private Configuration configuration;

  @SuppressWarnings("unchecked")
  @Before
  public void before() throws IOException {
    configuration = new Configuration();
    seqFilesOutputPath = new Path("seqfiles");

    lucene2Seq = new SequenceFilesFromLuceneStorage();
    lucene2SeqConf = new LuceneStorageConfiguration(configuration,
      asList(getIndexPath()),
      seqFilesOutputPath,
      SingleFieldDocument.ID_FIELD,
      asList(SingleFieldDocument.FIELD));

    document1 = new SingleFieldDocument("1", "This is test document 1");
    document2 = new SingleFieldDocument("2", "This is test document 2");
    document3 = new SingleFieldDocument("3", "This is test document 3");
  }

  @After
  public void after() throws IOException {
    HadoopUtil.delete(lucene2SeqConf.getConfiguration(), lucene2SeqConf.getSequenceFilesOutputPath());
    HadoopUtil.delete(lucene2SeqConf.getConfiguration(), lucene2SeqConf.getIndexPaths());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testRun() throws Exception {
    commitDocuments(document1, document2, document3);

    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();

    assertSimpleDocumentEquals(document1, iterator.next());
    assertSimpleDocumentEquals(document2, iterator.next());
    assertSimpleDocumentEquals(document3, iterator.next());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testRun_skipEmptyIdFieldDocs() throws IOException {
    commitDocuments(document1, new SingleFieldDocument("", "This is a test document with no id"), document2);

    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();

    assertSimpleDocumentEquals(document1, iterator.next());
    assertSimpleDocumentEquals(document2, iterator.next());
    assertFalse(iterator.hasNext());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testRun_skipEmptyFieldDocs() throws IOException {
    commitDocuments(document1, new SingleFieldDocument("4", ""), document2);

    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();

    assertSimpleDocumentEquals(document1, iterator.next());
    assertSimpleDocumentEquals(document2, iterator.next());
    assertFalse(iterator.hasNext());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testRun_skipUnstoredFields() throws IOException {
    commitDocuments(new UnstoredFieldsDocument("5", "This is test document 5"));

    lucene2SeqConf = new LuceneStorageConfiguration(configuration,
      asList(getIndexPath()),
      seqFilesOutputPath,
      SingleFieldDocument.ID_FIELD,
      asList(UnstoredFieldsDocument.FIELD, UnstoredFieldsDocument.UNSTORED_FIELD));

    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();

    assertFalse(iterator.next().getSecond().toString().contains("null"));
    assertFalse(iterator.hasNext());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testRun_maxHits() throws IOException {
    commitDocuments(document1, document2, document3, new SingleFieldDocument("4", "This is test document 4"));

    lucene2SeqConf.setMaxHits(3);
    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();

    assertSimpleDocumentEquals(document1, iterator.next());
    assertSimpleDocumentEquals(document2, iterator.next());
    assertSimpleDocumentEquals(document3, iterator.next());
    assertFalse(iterator.hasNext());
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testRun_query() throws IOException {
    commitDocuments(document1, document2, document3, new SingleFieldDocument("4", "Mahout is cool"));

    Query query = new TermQuery(new Term(lucene2SeqConf.getFields().get(0), "mahout"));

    lucene2SeqConf.setQuery(query);
    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();

    assertSimpleDocumentEquals(new SingleFieldDocument("4", "Mahout is cool"), iterator.next());
    assertFalse(iterator.hasNext());
  }

  @Test
  public void testRun_multipleFields() throws IOException {
    lucene2SeqConf = new LuceneStorageConfiguration(configuration,
      asList(getIndexPath()),
      seqFilesOutputPath,
      SingleFieldDocument.ID_FIELD,
      asList(MultipleFieldsDocument.FIELD, MultipleFieldsDocument.FIELD1, MultipleFieldsDocument.FIELD2));

    MultipleFieldsDocument multipleFieldsDocument1 = new MultipleFieldsDocument("1", "This is field 1-1", "This is field 1-2", "This is field 1-3");
    MultipleFieldsDocument multipleFieldsDocument2 = new MultipleFieldsDocument("2", "This is field 2-1", "This is field 2-2", "This is field 2-3");
    MultipleFieldsDocument multipleFieldsDocument3 = new MultipleFieldsDocument("3", "This is field 3-1", "This is field 3-2", "This is field 3-3");
    commitDocuments(multipleFieldsDocument1, multipleFieldsDocument2, multipleFieldsDocument3);

    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();

    assertMultipleFieldsDocumentEquals(multipleFieldsDocument1, iterator.next());
    assertMultipleFieldsDocumentEquals(multipleFieldsDocument2, iterator.next());
    assertMultipleFieldsDocumentEquals(multipleFieldsDocument3, iterator.next());
  }

  @Test
  public void testRun_numericField() throws IOException {
    lucene2SeqConf = new LuceneStorageConfiguration(configuration,
      asList(getIndexPath()),
      seqFilesOutputPath,
      SingleFieldDocument.ID_FIELD,
      asList(NumericFieldDocument.FIELD, NumericFieldDocument.NUMERIC_FIELD));

    NumericFieldDocument doc1 = new NumericFieldDocument("1", "This is field 1", 100);
    NumericFieldDocument doc2 = new NumericFieldDocument("2", "This is field 2", 200);
    NumericFieldDocument doc3 = new NumericFieldDocument("3", "This is field 3", 300);

    commitDocuments(doc1, doc2, doc3);

    lucene2Seq.run(lucene2SeqConf);

    Iterator<Pair<Text, Text>> iterator = lucene2SeqConf.getSequenceFileIterator();

    assertNumericFieldEquals(doc1, iterator.next());
    assertNumericFieldEquals(doc2, iterator.next());
    assertNumericFieldEquals(doc3, iterator.next());
  }
}
