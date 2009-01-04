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

package org.apache.mahout.cf.taste.example.bookcrossing;

import org.apache.mahout.cf.taste.impl.model.GenericItem;

final class Book extends GenericItem<String> {

  private final String isbn;
  private final String title;
  private final String author;
  private final int year;
  private final String publisher;

  Book(String isbn, String title, String author, int year, String publisher) {
    super(isbn);
    this.isbn = isbn;
    this.title = title;
    this.author = author;
    this.year = year;
    this.publisher = publisher;
  }


  String getIsbn() {
    return isbn;
  }

  String getTitle() {
    return title;
  }

  String getAuthor() {
    return author;
  }

  int getYear() {
    return year;
  }

  String getPublisher() {
    return publisher;
  }

}