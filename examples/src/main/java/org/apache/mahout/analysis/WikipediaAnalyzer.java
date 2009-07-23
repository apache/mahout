package org.apache.mahout.analysis;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopAnalyzer;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardFilter;
import org.apache.lucene.wikipedia.analysis.WikipediaTokenizer;

import java.io.Reader;


/**
 *
 *
 **/
public class WikipediaAnalyzer extends Analyzer {

  private CharArraySet stopSet;

  public WikipediaAnalyzer() {
    stopSet = (CharArraySet) StopFilter.makeStopSet(StopAnalyzer.ENGLISH_STOP_WORDS);
  }

  public WikipediaAnalyzer(CharArraySet stopSet) {
    this.stopSet = stopSet;
  }

  @Override
  public TokenStream tokenStream(String fieldName, Reader reader) {
    TokenStream result = new WikipediaTokenizer(reader);
    result = new StandardFilter(result);
    result = new LowerCaseFilter(result);
    result = new StopFilter(true, result, stopSet);


    return result;
  }
}
