package org.apache.mahout.utils.vectors.io;

import org.apache.mahout.utils.vectors.TermInfo;

import java.io.IOException;


/**
 *
 *
 **/
public interface TermInfoWriter {

  public void write(TermInfo ti) throws IOException;

  public void close() throws IOException;
}
