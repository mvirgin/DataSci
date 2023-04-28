### Matthew Virgin
### Dr. Chaofan Chen
### COS 482
### 24 April 2023

#### Homework 3, Task 2

from pyspark.sql import SparkSession
import findspark
findspark.init()

spark = SparkSession.builder.master("local").appName("myapp").getOrCreate()
sc = spark.sparkContext

### a
data_rdd = sc.parallelize([(0,0.46), (2, 0.4485), (3, 0.053), (1, 0.0375)])
data = spark.createDataFrame(data_rdd, ['id', 'val'])

### b
query = spark.sql("SELECT val FROM data WHERE data.id=2")
print(query.collect())

### c
query2 = spark.sql("SELECT MAX(val) FROM data")

spark.stop()

##