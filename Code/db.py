import sqlite3


class DataSet(object):

    def __init__(self, database):
        self.database = database
        self.connection = None

    def connect(self):
        if self.database is None:
            return False
        conn = sqlite3.connect(self.database)
        self.connection = conn

    def insert_rows(self, rows):
        cursor = self.connection.cursor()
        cursor.executemany('''INSERT OR IGNORE INTO dataset(routing, delay, lambda, nodes)
                  VALUES(?,?,?,?)''', rows)
        self.connection.commit()

    def create_table(self):
        cursor = self.connection.cursor()
        cursor.execute('''CREATE TABLE  IF NOT EXISTS dataset
             (routing text, delay text UNIQUE, lambda integer, nodes integer)''')
        self.connection.commit()

    def select(self, nodes):
        cursor = self.connection.cursor()
        test_x = []
        test_y = []
        for row in cursor.execute('SELECT * FROM dataset WHERE nodes=?', (nodes,)):
            print(row)
            test_x.append(row[1])
            test_y.append(row[2])
        cursor.close()
        return test_x, test_y
