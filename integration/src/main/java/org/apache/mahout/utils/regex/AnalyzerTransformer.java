package org.apache.mahout.utils.regex;


import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.lucene.TokenStreamIterator;

import java.io.IOException;
import java.io.StringReader;

/**
 *
 *
 **/
public class AnalyzerTransformer implements RegexTransformer {
  protected Analyzer analyzer;
  protected String fieldName = "text";

  public AnalyzerTransformer() {
    this(new StandardAnalyzer(Version.LUCENE_34), "text");
  }

  public AnalyzerTransformer(Analyzer analyzer) {
    this(analyzer, "text");
  }

  public AnalyzerTransformer(Analyzer analyzer, String fieldName) {
    this.analyzer = analyzer;
    this.fieldName = fieldName;
  }

  @Override
  public String transformMatch(String match) {
    StringBuilder result = new StringBuilder();
    try {
      TokenStream ts = analyzer.reusableTokenStream(fieldName, new StringReader(match));
      ts.addAttribute(CharTermAttribute.class);
      TokenStreamIterator iter = new TokenStreamIterator(ts);
      while (iter.hasNext()) {
        result.append(iter.next()).append(" ");
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return result.toString();
  }

  public Analyzer getAnalyzer() {
    return analyzer;
  }

  public void setAnalyzer(Analyzer analyzer) {
    this.analyzer = analyzer;
  }
}
