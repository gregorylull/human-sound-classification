import sqlalchemy as db
from sqlalchemy.ext.declarative import declarative_base

params = {
    'host': '35.167.147.109',
    'user': 'ubuntu',
    'port': 5432
}

connection_string = f'postgres://ubuntu:{params["host"]}@{params["host"]}:{params["port"]}/store'
engine = db.create_engine(connection_string)
connection = engine.connect()

Base = declarative_base()
class AllStarFull(Base):
    __tablename__ = 'allstartfull'


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


    playerid = db.Column(db.VARCHAR(20), nullable=False, primary_key=True)
    yearid = db.Column(db.Integer, nullable=False, primary_key=True)
    gamenum = db.Column(db.VARCHAR(20), nullable=False, primary_key=True)

    gameid = db.Column(db.VARCHAR(12) )
    teamid = db.Column(db.String)
    lgid = db.Column(db.String)
    gp = db.Column(db.VARCHAR(20))
    startingpos = db.Column(db.VARCHAR(20))


class Salries(Base):
    __tablename__ = 'salaries'

    yearid = db.Column(db.Integer, nullable=False, primary_key=True)
    teamid = db.Column(db.VARCHAR(3), nullable=False, primary_key=True)
    lgid = db.Column(db.VARCHAR(2), nullable=False, primary_key=True)
    playerid = db.Column(db.VARCHAR(9))
    salary = db.Column(db.String)

    lgid = db.Column(db.String)
    gp = db.Column(db.VARCHAR(20))
    startingpos = db.Column(db.VARCHAR(20))