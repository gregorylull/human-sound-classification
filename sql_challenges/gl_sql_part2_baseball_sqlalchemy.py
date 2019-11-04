"""
# Challenge Set 9
## Part II: Baseball Data

*Introductory - Intermediate level SQL*

--

Please complete this exercise via SQLalchemy and Jupyter notebook.

We will be working with the Lahman baseball data we uploaded to your AWS instance in class. 

1. What was the total spent on salaries by each team, each year?
instances = session.query(Salaries.teamid, Salaries.yearid, func.sum(Salaries.salary).label('total_salary'))\
    .group_by(Salaries.teamid, Salaries.yearid)\
    .order_by(Salaries.teamid, Salaries.yearid)

for instance in instances:
    print(instance.teamid, instance.yearid, instance.total_salary)

ANA 1997 31135472.0
ANA 1998 41281000.0
ANA 1999 55388166.0
ANA 2000 51464167.0
ANA 2001 47535167.0
ANA 2002 61721667.0
ANA 2003 79031667.0
ANA 2004 100534667.0
ARI 1998 32347000.0
ARI 1999 68703999.0
ARI 2000 81027833.0
ARI 2001 85082999.0
...

2. What is the first and last year played for each player?
*Hint:* Create a new table from 'Fielding.csv'.

instances = session.query(Fielding.playerid, func.min(Fielding.yearid).label('start') , func.max(Fielding.yearid).label('end'))\
    .group_by(Fielding.playerid)\
    .order_by(Fielding.playerid)

for instance in instances[:50]:
    print(instance.playerid, instance.start, instance.end)

aardsda01 2004 2013
aaronha01 1954 1976
aaronto01 1962 1971
aasedo01 1977 1990
abadan01 2001 2003
abadfe01 2010 2013
abadijo01 1875 1875
abbated01 1897 1910
abbeybe01 1892 1896
abbeych01 1893 1897
...


3. Who has played the most all star games?
instances = session.query(AllStarFull.playerid, func.sum(db.cast(AllStarFull.gp, db.INTEGER)).label('game_played_total'))\
    .group_by(AllStarFull.playerid)\
    .order_by(db.desc('game_played_total'))

for instance in instances[:50]:
    print(instance.playerid, instance.game_played_total)

mayswi01 24
aaronha01 24
musiast01 24
ripkeca01 18
...


4. Which school has generated the most distinct players?
*Hint:* Create new table from 'CollegePlaying.csv'.

(glull) I don't see 'CollegePlaying.csv' in the .zip files.


5. Which players have the longest career?
Assume that the `debut` and `finalGame` columns comprise the start and end,
respectively, of a player's career.
*Hint:* Create a new table from 'Master.csv'.
Also note that strings can be converted to dates using the [`DATE`](https://wiki.postgresql.org/wiki/Working_with_Dates_and_Times_in_PostgreSQL#WORKING_with_DATETIME.2C_DATE.2C_and_INTERVAL_VALUES) function and can then be subtracted from each other yielding their difference in days.

instances = session.query(Master.playerid, (Master.finalgame - Master.debut).label('diff'))\
    .order_by(db.desc('diff'))

for instance in instances[:500]:
    print(instance.playerid, instance.diff)

altroni01 12862
orourji01 11836
minosmi01 11492
olearch01 11126
lathaar01 10678
mcguide01 10192
jennihu01 9954
eversjo01 9897
ryanno01 9873
streega01 9868
ansonca01 9647
moyerja01 9477
johnto01 9393
francju01 9278
...


6. What is the distribution of debut months?
*Hint:* Look at the `DATE` and [`EXTRACT`](https://www.postgresql.org/docs/current/static/functions-datetime.html#FUNCTIONS-DATETIME-EXTRACT) functions.
sub_months = session.query(Master.playerid, func.extract('month', Master.debut).label('debut_month')).subquery()

instances = session.query(sub_months.c.debut_month, func.count(sub_months.c.debut_month).label('count'))\
    .group_by(sub_months.c.debut_month)\
    .order_by(sub_months.c.debut_month)


for month, count in instances:
    print(month, count)

march 41
april 4711
may 2230
june 1893
july 1978
aug 1943
sept 5061
oct 308


7. What is the effect of table join order on mean salary for the players listed in the
main (master) table?
*Hint:* Perform two different queries, one that joins on playerID in the salary
table and other that joins on the same column in the master table.
You will have to use left joins for each since right joins are not currently supported with SQLalchemy.

(glull) i'm not sure what this question is asking for specifically, this is baseline:
SELECT avg(salary) FROM salaries;
1864357


(glull) modifying script from project 03 aws setup 03_HW_setup_baseball_database.md to create tables.
# Fielding.csv
# CollegePlaying.csv  ?? I don't see this anywhere
# Master.csv

CREATE TABLE IF NOT EXISTS Fielding (
    playerID varchar(255) NOT NULL,
    yearID varchar(255) NOT NULL,
    stint int DEFAULT NULL,
    teamID varchar(255) NOT NULL,
    lgID varchar(255) DEFAULT NULL,
    POS varchar(255) DEFAULT NULL,
    G int DEFAULT NULL,
    GS int DEFAULT NULL,
    InnOuts int DEFAULT NULL,
    PO int DEFAULT NULL,
    A int DEFAULT NULL,
    E int DEFAULT NULL,
    DP int DEFAULT NULL,
    PB int DEFAULT NULL,
    WP int DEFAULT NULL,
    SB int DEFAULT NULL,
    CS int DEFAULT NULL,
    ZR int DEFAULT NULL
);
COPY Fielding FROM '/home/ubuntu/baseballdata/Fielding.csv' DELIMITER ',' CSV HEADER;

CREATE TABLE IF NOT EXISTS Master (
    playerID varchar(255) NOT NULL,
    birthYear int DEFAULT NULL,
    birthMonth int DEFAULT NULL,
    birthDay int DEFAULT NULL,
    birthCountry varchar(255) DEFAULT NULL,
    birthState varchar(255) DEFAULT NULL,
    birthCity varchar(255) DEFAULT NULL,
    deathYear int DEFAULT NULL,
    deathMonth int DEFAULT NULL,
    deathDay int DEFAULT NULL,
    deathCountry varchar(255) DEFAULT NULL,
    deathState varchar(255) DEFAULT NULL,
    deathCity varchar(255) DEFAULT NULL,
    nameFirst varchar(255) DEFAULT NULL,
    nameLast varchar(255) DEFAULT NULL,
    nameGiven varchar(255) DEFAULT NULL,
    weight int DEFAULT NULL,
    height int DEFAULT NULL,
    bats varchar(255) DEFAULT NULL,
    throws varchar(255) DEFAULT NULL,
    debut DATE DEFAULT NULL,
    finalGame DATE DEFAULT NULL,
    retroID varchar(255) DEFAULT NULL,
    bbrefID varchar(255) DEFAULT NULL,
    PRIMARY KEY (playerID)
);
COPY Master FROM '/home/ubuntu/baseballdata/Master.csv' DELIMITER ',' CSV HEADER;



guides/tutorials
- get column names
    - result = session.exexcute('SELECT * FROM schools'); result.keys()
    - https://stackoverflow.com/questions/52251066/how-to-get-column-name-from-sqlalchemy/52252455

- execute raw sql query
    - `db.session.execute('<query>')`
    - https://stackoverflow.com/questions/17972020/how-to-execute-raw-sql-in-flask-sqlalchemy-app

- example connection and query https://www.pythoncentral.io/introductory-tutorial-python-sqlalchemy/

"""


import sqlalchemy as db
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

params = {
    'host': '35.167.147.109',
    'user': 'ubuntu',
    'port': 5432,
    'database': 'baseball'
}

connection_string = f'postgres://ubuntu:{params["host"]}@{params["host"]}:{params["port"]}/{params["database"]}'

Base = declarative_base()
class AllStarFull(Base):
    __tablename__ = 'allstarfull'

    playerid = db.Column(db.VARCHAR(20), nullable=False, primary_key=True)
    yearid = db.Column(db.Integer, nullable=False, primary_key=True)
    gamenum = db.Column(db.VARCHAR(20), nullable=False, primary_key=True)
    gameid = db.Column(db.VARCHAR(12) )
    teamid = db.Column(db.String)
    lgid = db.Column(db.String)
    gp = db.Column(db.VARCHAR(20))
    startingpos = db.Column(db.VARCHAR(20))


class Schools(Base):
    __tablename__ = 'schools'

    schoolid = db.Column(db.VARCHAR(15), nullable=False, primary_key=True)
    schoolname = db.Column(db.VARCHAR(255))
    schoolcity = db.Column(db.VARCHAR(55))
    schoolstate = db.Column(db.VARCHAR(55))
    schoolnick = db.Column(db.VARCHAR(55))

class Salaries(Base):
    __tablename__ = 'salaries'

    yearid = db.Column(db.Integer, nullable=False, primary_key=True)
    teamid = db.Column(db.VARCHAR(3), nullable=False, primary_key=True)
    lgid = db.Column(db.VARCHAR(2), nullable=False, primary_key=True)
    playerid = db.Column(db.VARCHAR(9))
    salary = db.Column(postgresql.DOUBLE_PRECISION)

class Master(Base):
    __tablename__ = 'master'

    playerid = db.Column(db.VARCHAR(255), nullable=False, primary_key=True)
    birthyear = db.Column(db.Integer)
    birthmonth = db.Column(db.Integer)
    birthday = db.Column(db.Integer)
    birthcountry = db.Column(db.VARCHAR(255))
    birthstate = db.Column(db.VARCHAR(255))
    birthcity = db.Column(db.VARCHAR(255))
    deathyear = db.Column(db.Integer)
    deathmonth = db.Column(db.Integer)
    deathday = db.Column(db.Integer)
    deathcountry = db.Column(db.VARCHAR(255))
    deathstate = db.Column(db.VARCHAR(255))
    deathcity = db.Column(db.VARCHAR(255))
    namefirst = db.Column(db.VARCHAR(255))
    namelast = db.Column(db.VARCHAR(255))
    namegiven = db.Column(db.VARCHAR(255))
    weight = db.Column(db.Integer)
    height = db.Column(db.Integer)
    bats = db.Column(db.VARCHAR(255))
    throws = db.Column(db.VARCHAR(255))
    debut = db.Column(db.DATE)
    finalgame = db.Column(db.DATE)
    retroid = db.Column(db.VARCHAR(255))
    bbrefid = db.Column(db.VARCHAR(255))

class Fielding(Base):
    __tablename__ = 'fielding'

    playerid = db.Column(db.VARCHAR(255), nullable=False, primary_key=True)
    yearid = db.Column(db.VARCHAR(255), nullable=False)
    stint = db.Column(db.Integer)
    teamid = db.Column(db.VARCHAR(255), nullable=False)
    lgid = db.Column(db.VARCHAR(255))
    pos = db.Column(db.VARCHAR(255))
    g = db.Column(db.Integer)
    gs = db.Column(db.Integer)
    innouts = db.Column(db.Integer)
    po = db.Column(db.Integer)
    a = db.Column(db.Integer)
    e = db.Column(db.Integer)
    dp = db.Column(db.Integer)
    pb = db.Column(db.Integer)
    wp = db.Column(db.Integer)
    sb = db.Column(db.Integer)
    cs = db.Column(db.Integer)
    zr = db.Column(db.Integer)

# this command will create a table, but we already have tables
# Base.metadata.create_all(engine)

def connect_to_db():
    engine = db.create_engine(connection_string, pool_size=5)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    return DBSession

DBSession = connect_to_db()
session = DBSession()

# test connection
print('\ntesting connection\n')
for instance in session.query(Schools)[:5]:
    print(instance.schoolid, instance.schoolname)


# question 1 total salary per team per year
print('\nQuestion 1 total salary per team per year\n')
instances = session.query(Salaries.teamid, Salaries.yearid, func.sum(Salaries.salary).label('total_salary'))\
    .group_by(Salaries.teamid, Salaries.yearid)\
    .order_by(Salaries.teamid, Salaries.yearid)

for instance in instances:
    print(instance.teamid, instance.yearid, instance.total_salary)


# question 2 player first and last year
print('\nQuestion 2 player first and last year\n')
instances = session.query(Fielding.playerid, func.min(Fielding.yearid).label('start') , func.max(Fielding.yearid).label('end'))\
    .group_by(Fielding.playerid)\
    .order_by(Fielding.playerid)

for instance in instances[:50]:
    print(instance.playerid, instance.start, instance.end)


# question 3 who has played the most of all star games
print('\nQuestion 3 who has played the most all star\n')
instances = session.query(AllStarFull.playerid, func.sum(db.cast(AllStarFull.gp, db.INTEGER)).label('game_played_total'))\
    .group_by(AllStarFull.playerid)\
    .order_by(db.desc('game_played_total'))

for instance in instances[:50]:
    print(instance.playerid, instance.game_played_total)


# question 5 which players have the longest career?
print('\nQuestion 5 longest career\n')
instances = session.query(Master.playerid, (Master.finalgame - Master.debut).label('diff'))\
    .order_by(db.desc('diff'))

for instance in instances[:500]:
    print(instance.playerid, instance.diff)


# 6. What is the distribution of debut months?
print('\nQuestion 6 distribution of debut months\n')
sub_months = session.query(Master.playerid, func.extract('month', Master.debut).label('debut_month')).subquery()

instances = session.query(sub_months.c.debut_month, func.count(sub_months.c.debut_month).label('count'))\
    .group_by(sub_months.c.debut_month)\
    .order_by(sub_months.c.debut_month)


for month, count in instances:
    print(month, count)

# 7. What is the effect of table join order on mean salary for the players listed in the
# main (master) table?
# *Hint:* Perform two different queries, one that joins on playerID in the salary
# table and other that joins on the same column in the master table.
# You will have to use left joins for each since right joins are not currently supported with SQLalchemy.
#func.avg(Salaries).label('salary_avg')
instances = session.query(Salaries)\
    .outerjoin(Salaries, Salaries.playerid == Master.playerid)

for salary_avg in instances:
    print(salary_avg)
