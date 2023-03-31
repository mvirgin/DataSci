import psycopg2

conn = psycopg2.connect("host=localhost dbname=moviesdb user=postgres \
                        password=cos482!hw2")

curr = conn.cursor()

### create tables

try:
    curr.execute("""
        CREATE TABLE Person(
            id integer NOT NULL PRIMARY KEY,
            fname varchar(30),
            lname varchar(30),
            gender varchar(15)
        )
    """)

    curr.execute("""
        CREATE TABLE Movie(
            id integer NOT NULL PRIMARY KEY,
            name text,
            year integer,
            rank float
        )
    """)

    curr.execute("""
        CREATE TABLE ActsIn(
            pid integer NOT NULL,
            mid integer NOT NULL,
            role text,
            PRIMARY KEY (pid, mid, role),
            CONSTRAINT f_key_pid FOREIGN KEY (pid) REFERENCES Person(id)
                ON UPDATE CASCADE
                ON DELETE RESTRICT,
            CONSTRAINT f_key_mid FOREIGN KEY (mid) REFERENCES Movie(id)
                ON UPDATE CASCADE
                ON DELETE RESTRICT
        )
    """)

    curr.execute("""
        CREATE TABLE Director(
            id integer NOT NULL PRIMARY KEY,
            fname varchar(30),
            lname varchar(30)
        )
    """)

    curr.execute("""
        CREATE TABLE Directs(
            did integer NOT NULL,
            mid integer NOT NULL,
            PRIMARY KEY (did, mid),
            CONSTRAINT f_key_did FOREIGN KEY (did) REFERENCES Director(id)
                ON UPDATE CASCADE
                ON DELETE RESTRICT,
            CONSTRAINT f_key_mid FOREIGN KEY (mid) REFERENCES Movie(id)
                ON UPDATE CASCADE
                ON DELETE RESTRICT
        )
    """)

    conn.commit()
except:
    conn.rollback()

### loop over files and insert data into database

## inserting person (actors) into database:
pFile = open("IMDBPerson.txt", "r")
next(pFile)                             # skip first line explaining layout
for line in pFile:
    data = line.strip().split(",")

    id = int(data[0])
    fname = data[1]
    lname = data[2]
    gender = data[3]

    try:
        curr.execute("INSERT INTO Person VALUES (%s, %s, %s, %s)",
                     (id, fname, lname, gender))
        conn.commit()

    ## Removed IntegrityConstraintViolation from below because it wasn't
    ## working for some reason?    
    except:
        conn.rollback()

pFile.close()

## inserting movies into database:
mFile = open("IMDBMovie.txt", "r")
next(mFile)
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

    try:
        curr.execute("INSERT INTO Movie VALUES (%s, %s, %s, %s)",
                     (id, name, year, rank))
        conn.commit()
    
    except:
        conn.rollback()
    
mFile.close()

## inserting directors into database:
dFile = open("IMDBDirectors.txt", "r")
next(dFile)
for line in dFile:
    data = line.strip().split(",")

    id = int(data[0])
    fname = data[1]
    lname = data[2]

    try:
        curr.execute("INSERT INTO Director VALUES (%s, %s, %s)",
                     (id, fname, lname))
        conn.commit()
 
    except: 
        conn.rollback()

dFile.close()

## inserting roles, etc into ActsIn:
aFile = open("IMDBCast.txt", "r", encoding='latin-1')
next(aFile)
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

    try:
        curr.execute("INSERT INTO ActsIn VALUES (%s, %s, %s)",
                     (pid, mid, role))
        conn.commit()
 
    except: 
        conn.rollback()

aFile.close()

dsFile = open("IMDBMovie_Directors.txt", "r")
next(dsFile)
for line in dsFile:
    data = line.strip().split(",")

    did = int(data[0])
    mid = int(data[1])

    try:
        curr.execute("INSERT INTO Directs VALUES (%s, %s)", (did, mid))
        conn.commit()
    except:
        conn.rollback()
    
dsFile.close()

curr.close()
conn.close()