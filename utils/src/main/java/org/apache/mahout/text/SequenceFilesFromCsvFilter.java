package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.FileLineIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Implements an example csv to sequence file parser.
 */
public final class SequenceFilesFromCsvFilter extends SequenceFilesFromDirectoryFilter {
  private static final Logger log = LoggerFactory.getLogger(SequenceFilesFromCsvFilter.class);
  private static final Pattern TAB = Pattern.compile("\\t");

  public static final String[] KEY_COLUMN_OPTION = {"keyColumn", "kcol"};
  public static final String[] VALUE_COLUMN_OPTION = {"valueColumn", "vcol"};

  private volatile int keyColumn;
  private volatile int valueColumn;

  private SequenceFilesFromCsvFilter() {
    // not initializing anything here.
  }

  public SequenceFilesFromCsvFilter(Configuration conf, String keyPrefix, Map<String, String> options, ChunkedWriter writer)
    throws IOException {
    super(conf, keyPrefix, options, writer);
    this.keyColumn = Integer.parseInt(options.get(KEY_COLUMN_OPTION[0]));
    this.valueColumn = Integer.parseInt(options.get(VALUE_COLUMN_OPTION[0]));
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new SequenceFilesFromCsvFilter(), args);
  }

  @Override
  public void addOptions() {
    super.addOptions();
    addOption(KEY_COLUMN_OPTION[0], KEY_COLUMN_OPTION[1],
      "The key column. Default to 0", "0");
    addOption(VALUE_COLUMN_OPTION[0], VALUE_COLUMN_OPTION[1],
      "The value column. Default to 1", "1");
  }

  @Override
  public Map<String, String> parseOptions() throws IOException {
    Map<String, String> options = super.parseOptions();
    options.put(SequenceFilesFromDirectory.FILE_FILTER_CLASS_OPTION[0], this.getClass().getName());
    options.put(KEY_COLUMN_OPTION[0], getOption(KEY_COLUMN_OPTION[0]));
    options.put(VALUE_COLUMN_OPTION[0], getOption(VALUE_COLUMN_OPTION[0]));
    return options;
  }

  @Override
  protected final void process(FileStatus fst, Path current) throws IOException {
    if (fst.isDir()) {
      fs.listStatus(fst.getPath(),
                    new SequenceFilesFromCsvFilter(conf, prefix + Path.SEPARATOR + current.getName(),
                        this.options, writer));
    } else {
      StringBuilder file = new StringBuilder();
      InputStream in = fs.open(fst.getPath());
      for (String aFit : new FileLineIterable(in, charset, false)) {
        String[] columns = TAB.split(aFit);
        log.info("key : " + columns[keyColumn] + ", value : "
            + columns[valueColumn]);
        String key = columns[keyColumn];
        String value = columns[valueColumn];
        writer.write(prefix + key, value);
      }
    }
  }
}
