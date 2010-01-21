package org.apache.mahout.clustering.dirichlet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.store.LockObtainFailedException;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.Version;
import org.apache.mahout.clustering.dirichlet.models.L1ModelDistribution;
import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.utils.vectors.TFIDF;
import org.apache.mahout.utils.vectors.TermInfo;
import org.apache.mahout.utils.vectors.Weight;
import org.apache.mahout.utils.vectors.lucene.CachedTermInfo;
import org.apache.mahout.utils.vectors.lucene.LuceneIterable;
import org.apache.mahout.utils.vectors.lucene.TFDFMapper;
import org.apache.mahout.utils.vectors.lucene.VectorMapper;
import org.junit.After;
import org.junit.Before;


public class TestL1ModelClustering extends MahoutTestCase {

  @SuppressWarnings("unchecked")
  private class MapElement implements Comparable {

    public MapElement(double pdf, String doc) {
      super();
      this.pdf = pdf;
      this.doc = doc;
    }

    Double pdf;

    String doc;

    @Override
    // reverse compare to sort in reverse order
    public int compareTo(Object o) {
      MapElement e = (MapElement) o;
      if (e.pdf > pdf)
        return 1;
      else if (e.pdf < pdf)
        return -1;
      else
        return 0;
    }

    public String toString() {
      return pdf.toString() + " " + doc.toString();
    }

  }

  private static final String[] DOCS = { "The quick red fox jumped over the lazy brown dogs.",
      "The quick brown fox jumped over the lazy red dogs.", "The quick red cat jumped over the lazy brown dogs.",
      "The quick brown cat jumped over the lazy red dogs.", "Mary had a little lamb whose fleece was white as snow.",
      "Moby Dick is a story of a whale and a man obsessed.", "The robber wore a black fleece jacket and a baseball cap.",
      "The English Springer Spaniel is the best of all dogs." };

  private List<VectorWritable> sampleData;

  private RAMDirectory directory;

  private static final String[] DOCS2 = { "The quick red fox jumped over the lazy brown dogs.",
      "The quick brown fox jumped over the lazy red dogs.", "The quick red cat jumped over the lazy brown dogs.",
      "The quick brown cat jumped over the lazy red dogs.", "Mary had a little lamb whose fleece was white as snow.",
      "Mary had a little goat whose fleece was white as snow.", "Mary had a little lamb whose fleece was black as tar.",
      "Dick had a little goat whose fleece was white as snow.", "Moby Dick is a story of a whale and a man obsessed.",
      "Moby Bob is a story of a walrus and a man obsessed.", "Moby Dick is a story of a whale and a crazy man.",
      "The robber wore a black fleece jacket and a baseball cap.", "The robber wore a red fleece jacket and a baseball cap.",
      "The robber wore a white fleece jacket and a baseball cap.", "The English Springer Spaniel is the best of all dogs." };

  @Before
  protected void setUp() throws Exception {
    super.setUp();
    RandomUtils.useTestSeed();
  }

  private void getSampleData(String[] docs2) throws CorruptIndexException, LockObtainFailedException, IOException {
    sampleData = new ArrayList<VectorWritable>();
    directory = new RAMDirectory();
    IndexWriter writer = new IndexWriter(directory, new StandardAnalyzer(Version.LUCENE_CURRENT), true,
        IndexWriter.MaxFieldLength.UNLIMITED);
    for (int i = 0; i < docs2.length; i++) {
      Document doc = new Document();
      Field id = new Field("id", "doc_" + i, Field.Store.YES, Field.Index.NOT_ANALYZED_NO_NORMS);
      doc.add(id);
      //Store both position and offset information
      Field text = new Field("content", docs2[i], Field.Store.NO, Field.Index.ANALYZED, Field.TermVector.YES);
      doc.add(text);
      writer.addDocument(doc);
    }
    writer.close();
    IndexReader reader = IndexReader.open(directory, true);
    Weight weight = new TFIDF();
    TermInfo termInfo = new CachedTermInfo(reader, "content", 1, 100);
    VectorMapper mapper = new TFDFMapper(reader, weight, termInfo);
    LuceneIterable iterable = new LuceneIterable(reader, "id", "content", mapper);

    int i = 0;
    for (Vector vector : iterable) {
      assertNotNull(vector);
      System.out.println("Vector[" + i++ + "]=" + formatVector(vector));
      sampleData.add(new VectorWritable(vector));
    }
  }

  @After
  protected void tearDown() throws Exception {
  }

  private String formatVector(Vector v) {
    StringBuilder buf = new StringBuilder();
    int nzero = 0;
    Iterator<Element> iterateNonZero = v.iterateNonZero();
    while (iterateNonZero.hasNext()) {
      iterateNonZero.next();
      nzero++;
    }
    buf.append("(").append(nzero);
    buf.append("nz) [");
    int nextIx = 0;
    if (v != null) {
      // handle sparse Vectors gracefully, suppressing zero values
      for (int i = 0; i < v.size(); i++) {
        double elem = v.get(i);
        if (elem == 0.0)
          continue;
        if (i > nextIx)
          buf.append("..{").append(i).append("}=");
        buf.append(String.format("%.2f", elem)).append(", ");
        nextIx = i + 1;
      }
    }
    buf.append("]");
    return buf.toString();
  }

  private void printSamples(List<Model<VectorWritable>[]> result, int significant) {
    int row = 0;
    for (Model<VectorWritable>[] r : result) {
      int sig = 0;
      for (Model<VectorWritable> model : r) {
        if (model.count() > significant) {
          sig++;
        }
      }
      System.out.print("sample[" + row++ + "] (" + sig + ")= ");
      for (Model<VectorWritable> model : r) {
        if (model.count() > significant) {
          System.out.print(model.toString() + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }

  private void printClusters(Model<VectorWritable>[] models, List<VectorWritable> samples, String[] docs) {
    for (int m = 0; m < models.length; m++) {
      Model<VectorWritable> model = models[m];
      int count = model.count();
      if (count == 0)
        continue;
      System.out.println("Model[" + m + "] had " + count + " hits (!) and " + (samples.size()-count) + " misses (? in pdf order) during the last iteration:");
      MapElement[] map = new MapElement[samples.size()];
      // sort the samples by pdf
      for (int i = 0; i < samples.size(); i++) {
        VectorWritable sample = samples.get(i);
        map[i] = new MapElement(model.pdf(sample), docs[i]);
      }
      Arrays.sort(map);
      // now find the n=model.count() most likely docs and output them
      for (int i = 0; i < map.length; i++) {
        if (i < count)
          System.out.print("! ");
        else
          System.out.print("? ");
        System.out.println(map[i].doc);
      }
    }
  }

  public void testDocs() throws Exception {
    System.out.println("testDocs");
    getSampleData(DOCS);
    DirichletClusterer<VectorWritable> dc = new DirichletClusterer<VectorWritable>(sampleData, new L1ModelDistribution(sampleData
        .get(0)), 1.0, 15, 1, 0);
    List<Model<VectorWritable>[]> result = dc.cluster(10);
    assertNotNull(result);
    printSamples(result, 0);
    printClusters(result.get(result.size() - 1), sampleData, DOCS);
  }

  public void testDocs2() throws Exception {
    System.out.println("testDocs2");
    getSampleData(DOCS2);
    DirichletClusterer<VectorWritable> dc = new DirichletClusterer<VectorWritable>(sampleData, new L1ModelDistribution(sampleData
        .get(0)), 1.0, 15, 1, 0);
    List<Model<VectorWritable>[]> result = dc.cluster(10);
    assertNotNull(result);
    printSamples(result, 0);
    printClusters(result.get(result.size() - 1), sampleData, DOCS2);
  }

}
