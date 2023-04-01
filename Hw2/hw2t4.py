### Matthew Virgin
### Dr. Chaofan Chen
### COS 482
### 5 April 2023

#### Homework 2, task 4:

import psycopg2
import csv

conn = psycopg2.connect("host=localhost dbname=moviesdb user=postgres \
                        password=cos482!hw2")

curr = conn.cursor()

### finds the top k movies with the highest ratings made after start_year and
### before end_year. Stores these movies in a csv delimited by ; instead of ,
def find_best_movies_in_years(k, start_year, end_year):
    try:
        curr.execute("""SELECT * FROM Movie
                        WHERE year >= {start_year}
                        AND year <= {end_year}
                        ORDER BY rank DESC NULLS LAST
                        LIMIT {k}
                    """.format(start_year=start_year,end_year=end_year,k=k))
        movieRange = curr.fetchall()

        fileName = "{start_year}-{end_year}top{k}.csv".format(
            start_year=start_year,end_year=end_year,k=k)
        csvFile = open(fileName, mode = 'w', newline = '')
        writer = csv.writer(csvFile, delimiter = ';')

        for tuple in movieRange:
            writer.writerow(tuple)

        csvFile.close()

    except:
        conn.rollback()

### finding top 20 (k) movies from 1995(start_year) to 2004(end_year)
find_best_movies_in_years(20, 1995, 2004)