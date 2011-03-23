package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.FileLineIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Map;

/**
 * Default parser for parsing text into sequence files.
 */
public final class PrefixAdditionFilter extends SequenceFilesFromDirectoryFilter {
  private static final Logger log = LoggerFactory.getLogger(PrefixAdditionFilter.class);

  public PrefixAdditionFilter(Configuration conf, String keyPrefix, Map<String, String> options, ChunkedWriter writer)
    throws IOException {
    super(conf, keyPrefix, options, writer);
  }

  @Override
  protected void process(FileStatus fst, Path current) throws IOException {
    if (fst.isDir()) {
      fs.listStatus(fst.getPath(),
                    new PrefixAdditionFilter(conf, prefix + Path.SEPARATOR + current.getName(),
                        options, writer));
    } else {
      InputStream in = fs.open(fst.getPath());

      StringBuilder file = new StringBuilder();
      for (String aFit : new FileLineIterable(in, charset, false)) {
        file.append(aFit).append('\n');
      }
      String name = current.getName().equals(fst.getPath().getName())
          ? current.getName()
          : current.getName() + Path.SEPARATOR + fst.getPath().getName();
      writer.write(prefix + Path.SEPARATOR + name, file.toString());
    }
  }
}
