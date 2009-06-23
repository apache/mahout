package org.apache.mahout.utils.vectors.io;

import org.apache.mahout.utils.vectors.TermEntry;
import org.apache.mahout.utils.vectors.TermInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Writer;
import java.util.Iterator;


/**
 * Write ther TermInfo out to a {@link java.io.Writer}
 *
 **/
public class JWriterTermInfoWriter implements TermInfoWriter {
  private transient static Logger log = LoggerFactory.getLogger(JWriterTermInfoWriter.class);

  protected Writer writer;
  protected String delimiter;
  protected String field;

  public JWriterTermInfoWriter(Writer writer, String delimiter, String field) {
    this.writer = writer;
    this.delimiter = delimiter;
    this.field = field;
  }

  @Override
  public void write(TermInfo ti) throws IOException {

    Iterator<TermEntry> entIter = ti.getAllEntries();

    writer.write(String.valueOf(ti.totalTerms(field)));
    writer.write("\n");
    writer.write("#term" + delimiter + "doc freq" + delimiter + "idx");
    writer.write("\n");
    while (entIter.hasNext()) {
      TermEntry entry = entIter.next();
      writer.write(entry.term);
      writer.write(delimiter);
      writer.write(String.valueOf(entry.docFreq));
      writer.write(delimiter);
      writer.write(String.valueOf(entry.termIdx));
      writer.write("\n");
    }
    writer.flush();
    writer.close();
  }

  /**
   * Does NOT close the underlying writer
   * @throws IOException
   */
  public void close() throws IOException {

  }
}
