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

package org.apache.mahout.cf.taste.web;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import javax.servlet.ServletConfig;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

/**
 * <p>A servlet which returns recommendations, as its name implies. The servlet accepts GET and POST
 * HTTP requests, and looks for two parameters:</p>
 *
 * <ul>
 * <li><em>userID</em>: the user ID for which to produce recommendations</li>
 * <li><em>howMany</em>: the number of recommendations to produce</li>
 * <li><em>debug</em>: (optional) output a lot of information that is useful in debugging.
 * Defaults to false, of course.</li>
 * </ul>
 *
 * <p>The response is text, and contains a list of the IDs of recommended items, in descending
 * order of relevance, one per line.</p>
 *
 * <p>For example, you can get 10 recommendations for user 123 from the following URL (assuming
 * you are running taste in a web application running locally on port 8080):<br/>
 * {@code http://localhost:8080/taste/RecommenderServlet?userID=123&howMany=10}</p>
 *
 * <p>This servlet requires one {@code init-param} in {@code web.xml}: it must find
 * a parameter named "recommender-class" which is the name of a class that implements
 * {@link Recommender} and has a no-arg constructor. The servlet will instantiate and use
 * this {@link Recommender} to produce recommendations.</p>
 */
public final class RecommenderServlet extends HttpServlet {

  private static final int NUM_TOP_PREFERENCES = 20;
  private static final int DEFAULT_HOW_MANY = 20;

  private Recommender recommender;

  @Override
  public void init(ServletConfig config) throws ServletException {
    super.init(config);
    String recommenderClassName = config.getInitParameter("recommender-class");
    if (recommenderClassName == null) {
      throw new ServletException("Servlet init-param \"recommender-class\" is not defined");
    }
    RecommenderSingleton.initializeIfNeeded(recommenderClassName);
    recommender = RecommenderSingleton.getInstance().getRecommender();
  }

  @Override
  public void doGet(HttpServletRequest request,
                    HttpServletResponse response) throws ServletException {

    String userIDString = request.getParameter("userID");
    if (userIDString == null) {
      throw new ServletException("userID was not specified");
    }
    long userID = Long.parseLong(userIDString);
    String howManyString = request.getParameter("howMany");
    int howMany = howManyString == null ? DEFAULT_HOW_MANY : Integer.parseInt(howManyString);
    boolean debug = Boolean.parseBoolean(request.getParameter("debug"));
    String format = request.getParameter("format");
    if (format == null) {
      format = "text";
    }

    try {
      List<RecommendedItem> items = recommender.recommend(userID, howMany);
      if ("text".equals(format)) {
        writePlainText(response, userID, debug, items);
      } else if ("xml".equals(format)) {
        writeXML(response, items);
      } else if ("json".equals(format)) {
        writeJSON(response, items);
      } else {
        throw new ServletException("Bad format parameter: " + format);
      }
    } catch (TasteException te) {
      throw new ServletException(te);
    } catch (IOException ioe) {
      throw new ServletException(ioe);
    }

  }

  private static void writeXML(HttpServletResponse response, Iterable<RecommendedItem> items) throws IOException {
    response.setContentType("application/xml");
    response.setCharacterEncoding("UTF-8");
    response.setHeader("Cache-Control", "no-cache");
    PrintWriter writer = response.getWriter();
    writer.print("<?xml version=\"1.0\" encoding=\"UTF-8\"?><recommendedItems>");
    for (RecommendedItem recommendedItem : items) {
      writer.print("<item><value>");
      writer.print(recommendedItem.getValue());
      writer.print("</value><id>");
      writer.print(recommendedItem.getItemID());
      writer.print("</id></item>");
    }
    writer.println("</recommendedItems>");
  }

  private static void writeJSON(HttpServletResponse response, Iterable<RecommendedItem> items) throws IOException {
    response.setContentType("application/json");
    response.setCharacterEncoding("UTF-8");
    response.setHeader("Cache-Control", "no-cache");
    PrintWriter writer = response.getWriter();
    writer.print("{\"recommendedItems\":{\"item\":[");
    boolean first = true;
    for (RecommendedItem recommendedItem : items) {
      if (first) {
        first = false;
      } else {
        writer.print(',');
      }
      writer.print("{\"value\":\"");
      writer.print(recommendedItem.getValue());
      writer.print("\",\"id\":\"");
      writer.print(recommendedItem.getItemID());
      writer.print("\"}");
    }
    writer.println("]}}");
  }

  private void writePlainText(HttpServletResponse response,
                              long userID,
                              boolean debug,
                              Iterable<RecommendedItem> items) throws IOException, TasteException {
    response.setContentType("text/plain");
    response.setCharacterEncoding("UTF-8");
    response.setHeader("Cache-Control", "no-cache");
    PrintWriter writer = response.getWriter();
    if (debug) {
      writeDebugRecommendations(userID, items, writer);
    } else {
      writeRecommendations(items, writer);
    }
  }

  private static void writeRecommendations(Iterable<RecommendedItem> items, PrintWriter writer) {
    for (RecommendedItem recommendedItem : items) {
      writer.print(recommendedItem.getValue());
      writer.print('\t');
      writer.println(recommendedItem.getItemID());
    }
  }

  private void writeDebugRecommendations(long userID, Iterable<RecommendedItem> items, PrintWriter writer)
    throws TasteException {
    DataModel dataModel = recommender.getDataModel();
    writer.print("User:");
    writer.println(userID);
    writer.print("Recommender: ");
    writer.println(recommender);
    writer.println();
    writer.print("Top ");
    writer.print(NUM_TOP_PREFERENCES);
    writer.println(" Preferences:");
    PreferenceArray rawPrefs = dataModel.getPreferencesFromUser(userID);
    int length = rawPrefs.length();
    PreferenceArray sortedPrefs = rawPrefs.clone();
    sortedPrefs.sortByValueReversed();
    // Cap this at NUM_TOP_PREFERENCES just to be brief
    int max = Math.min(NUM_TOP_PREFERENCES, length);
    for (int i = 0; i < max; i++) {
      Preference pref = sortedPrefs.get(i);
      writer.print(pref.getValue());
      writer.print('\t');
      writer.println(pref.getItemID());
    }
    writer.println();
    writer.println("Recommendations:");
    for (RecommendedItem recommendedItem : items) {
      writer.print(recommendedItem.getValue());
      writer.print('\t');
      writer.println(recommendedItem.getItemID());
    }
  }

  @Override
  public void doPost(HttpServletRequest request,
                     HttpServletResponse response) throws ServletException {
    doGet(request, response);
  }

  @Override
  public String toString() {
    return "RecommenderServlet[recommender:" + recommender + ']';
  }

}
