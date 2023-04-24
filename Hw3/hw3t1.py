### Matthew Virgin
### Dr. Chaofan Chen
### COS 482
### 24 April 2023

#### Homework 3, Task 1

from pymongo import MongoClient, HASHED
from bson.objectid import ObjectId
import random

### a
client = MongoClient('mongodb://localhost:27017/')

db = client["moviesdb"]     # create db called moviesdb

## create collection called data, shard on name
collection = db["data"]
collection.create_index([("name", HASHED)])

## inserting person (actors) into database:
pFile = open("IMDB/IMDBPerson.txt", "r")
next(pFile)                             # skip first line explaining layout
pList = []                              # create list of people documents
for line in pFile:
    data = line.strip().split(",")

    id = int(data[0])
    fname = data[1]
    lname = data[2]
    gender = data[3]

    person_doc = {
            "_id": ObjectId(),
            "pe_id": id,
            "fname": fname,
            "lname": lname,
            "gender": gender
        }

    pList.append(person_doc)
pFile.close()

people = collection.insert_many(pList)  # insert person docs into collection

str(ObjectId())[:18] + str(random.uniform(0,1))

## inserting movies into database:
mFile = open("IMDB/IMDBMovie.txt", "r")
next(mFile)
mList = []
for line in mFile:
    data = line.strip().split(",")
    
    if len(data) == 4:          # title has no comma
        id = int(data[0])
        name = data[1]
        year = int(data[2])
        rank = data[3]
        if rank:
            rank = float(data[3])
        else:
            rank = None
    else:                       # title has comma
        id = int(data[0])
        rank = data[-1]
        if rank:
            rank = float(data[-1])
        else:
            rank = None
        year = int(data[-2])
        name = str(data[1:-2]).replace('[','').replace(']','').replace("'",'')

    movie_doc = {
        "_id": ObjectId(),
        "mo_id": id,
        "name": name,
        "year": year,
        "rank": rank
    }
    mList.append(movie_doc)
mFile.close()

movies = collection.insert_many(mList)

str(ObjectId())[:18] + str(random.uniform(0,1))

## inserting directors into database:
dFile = open("IMDB/IMDBDirectors.txt", "r")
next(dFile)
dList = []
for line in dFile:
    data = line.strip().split(",")

    id = int(data[0])
    fname = data[1]
    lname = data[2]

    director_doc = {
        "_id": ObjectId(),
        "di_id": id,
        "fname": fname,
        "lname": lname
    }
    dList.append(director_doc)
dFile.close()

directors = collection.insert_many(dList)

## inserting roles, etc into ActsIn:
aFile = open("IMDB/IMDBCast.txt", "r", encoding='latin-1')
next(aFile)
aList = []
for line in aFile:
    data = line.strip().split(",")

    if len(data) == 3:
        pid = int(data[0])
        mid = int(data[1])
        role = data[2]
        if role:
            role = data[2]
        else:
            role = ' '          # not None so that role can be in pkey of ActsIn
    else:
        pid = int(data[0])
        mid = int(data[1])
        role = data[2:]
        if role:
            role = str(data[2:]).replace('[','').replace(']','').replace("'",'')
        else:
            role = None

    cast_doc = {
        "_id": ObjectId(),
        "pid": pid,
        "mid": mid,
        "role": role
    }
    aList.append(cast_doc)
aFile.close()

actsIn = collection.insert_many(aList)

dsFile = open("IMDB/IMDBMovie_Directors.txt", "r")
next(dsFile)
dsList = []
for line in dsFile:
    data = line.strip().split(",")

    did = int(data[0])
    mid = int(data[1])

    ds_doc = {
        "_id": ObjectId(),
        "did": did,
        "mid": mid
    }
    dsList.append(ds_doc)
dsFile.close()

directs = collection.insert_many(dsList)

### b
# search for name
# get mid from that
# search for mid
# get pid's from that, print each actors fname, lname, gender, and role
# get did from that, print each directors fname, lname
def fetchMovieInfo(movieName):
    ## get results for searching for a movie w name Shrek (2001)
    shrek_results = collection.find({"name": movieName}) 

    ## get all the movie ids (mids) out of those results and print each movies details
    shrek_mids = []
    for doc in shrek_results:
        print("Movie Info:")
        print(doc)
        print()
        shrek_mids.append(doc["mo_id"])

    ## query the database for those mids and store result of each query
    ## in a list of results. (It's possible for two movies to have the same name)
    mid_query_results = []              # list of cursors. Each cursor contains docs
    for i in range(len(shrek_mids)):
        mid_query_results.append(collection.find({"mid": shrek_mids[i]}))

    ## get the pid, roles, and dids from those queries
    shrek_pid_roles = {}        # dictionary storing pid's and their roles
    shrek_dids = []             # list of directors of shrek
    for result in range(len(mid_query_results)):
        for doc in mid_query_results[i]:
            ## if doc at pid exists
            try:
                ## update dictionary
                pid = doc["pid"]
                role = doc["role"]
                shrek_pid_roles[pid] = role
            except:     # if getting pid and role doesn't work, did is listed
                shrek_dids.append(doc["did"])

    ## for each item in the dict, query the database and print the persons info, 
    ## followed by their role
    print("Cast Info:")
    for key, value in shrek_pid_roles.items():
        person = collection.find_one({"pe_id": key}) # pid is unique, find doc
        print(key, person, value)
    print()

    ## query database for the directors of the movie, print them
    print("Director Info:")
    for d in shrek_dids:
        director = collection.find_one({"di_id": d})
        print(director)
    print()

fetchMovieInfo("Shrek (2001)")