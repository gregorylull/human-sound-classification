# glull soccer
"""

1. Which team scored the most points when playing at home?  

ANSWER: Real Madrid CF scored the most points.

SELECT 
    home_team_api_id, SUM(home_team_goal) as total_goals
FROM
    Match
GROUP BY
    home_team_api_id
ORDER BY
    total_goals DESC

home_team_api_id: 8633
sum of home_team_goal: 505

execute('SELECT * FROM Team WHERE team_api_id=?', (8633))


2. Did this team also score the most points when playing away?  

ANSWER: No, FC Barcelona (8634) won the most when playing away.

SELECT 
    away_team_api_id, SUM(away_team_goal) as total_goals
FROM
    Match
GROUP BY
    away_team_api_id
ORDER BY
    total_goals DESC

away_team_api_id: 8634,

print_rows(cursor.execute('SELECT * FROM Team WHERE team_api_id=?', (8634,)))

3. How many matches resulted in a tie?
ANSWER 6596 matches resulted in a tie

print_rows(cursor.execute('''
SELECT COUNT(DISTINCT(match_api_id)) FROM Match WHERE away_team_goal=home_team_goal
'''))

4. How many players have Smith for their last name?
How many have 'smith' anywhere in their name?
ANSWER: There are 15 players with Smith as their last name, and 18 players with smith anywhere.

print_rows(cursor.execute('''
SELECT COUNT(player_name)
FROM Player
WHERE
    player_name LIKE "% Smith%"
'''))

5. What was the median tie score?
Use the value determined in the previous question for the number of tie games.
*Hint:* PostgreSQL does not have a median function.
Instead, think about the steps required to calculate a median and use the
[`WITH`](https://www.postgresql.org/docs/8.4/static/queries-with.html)
command to store stepwise results as a table and then operate on these results.

ANSWER: the median tie score is 1 point.

print_rows(cursor.execute('''...
away_team_goal, COUNT(match_api_id)
0, 1978
1, 3014
2, 1310
3, 264
4, 27
5, 2
6, 1


print_rows(cursor.execute('''
WITH tie_games AS (
SELECT match_api_id, away_team_goal
FROM Match

WHERE away_team_goal=home_team_goal
GROUP BY
    match_api_id
), tie_games_hist AS (
    SELECT away_team_goal, COUNT(match_api_id)
    FROM tie_games
    GROUP BY away_team_goal
)

SELECT * FROM tie_games_hist

'''), [], None)


6. What percentage of players prefer their left or right foot?
*Hint:* Calculate either the right or left foot,
whichever is easier based on how you setup the problem.
ANSWER 26% prefer their left foot.
right: 8979
left: 3202

cursor.execute('''
SELECT COUNT(DISTINCT(player_api_id)) as total_right
FROM Player_Attributes
WHERE
    preferred_foot=?
''', ('right', ))



"""

import sqlite3
import numpy as np

conn = sqlite3.connect('soccer_database.sqlite')
conn.row_factory = sqlite3.Row

cursor = conn.cursor()

def select_all(table='Team', query='*'):
    return f'SELECT {query} FROM {table}'

def print_rows(cursor, columns=[], limit = 3):
    cols = columns or get_col(cursor)

    rows = cursor.fetchmany(limit) if limit else cursor.fetchall()

    print(', '.join(cols))
    for row in rows:
        results = get_row(row, cols)
        stringified = [str(result) for result in results]
        print(', '.join(stringified))

def get_col(cursor):
    descrip = cursor.description
    columns = []

    return [ column[0] for column in descrip]

def get_row(row, cols):
    result = []

    for col in cols:
        result.append(row[col])

    return result



    

# table: Team
# id
# team_api_id
# team_fifa_api_id
# team_long_name
# team_short_name


