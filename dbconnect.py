import psycopg2
from datetime import datetime

db = psycopg2.connect(host='103.252.1.150', dbname='postgres',user='postgres',password='postgres',port=5324)
# db = psycopg2.connect(host='localhost', dbname='postgres',user='postgres',password='postgres',port=5432)
cursor=db.cursor()

class Databases():
    def __init__(self):
        self.db = psycopg2.connect(host='103.252.1.150', dbname='postgres',user='postgres',password='postgres',port=5324)

        self.cursor = self.db.cursor()

    def __del__(self):
        self.db.close()
        self.cursor.close()

    def execute(self,query,args={}):
        self.cursor.execute(query,args)
        row = self.cursor.fetchall()
        return row

    def commit(self):
        self.cursor.commit()

class CRUD(Databases):
    def insertDB(self, schema, table, colum, data):
        sql = " INSERT INTO {schema}.{table}({colum}) VALUES ('{data}') ;".format(schema=schema, table=table,
                                                                                  colum=colum, data=data)
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(" insert DB err ", e)

    def save_sensor_data(self,t, temp, pH, DO, conductivity):
        self.cursor.execute(
            "INSERT INTO sensor_data (\"DATETIME\", \"PH\", \"TEMP\", \"DO\", \"conductivity\") VALUES (%s, %s, %s, %s, %s)",
            (t, pH, temp, DO, conductivity))
        self.db.commit()

    def readDB(self, schema, table, colum):
        sql = " SELECT {colum} from {schema}.{table}".format(colum=colum, schema=schema, table=table)
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
        except Exception as e:
            result = (" read DB err", e)

        return result

    def updateDB(self, schema, table, colum, value, condition):
        sql = " UPDATE {schema}.{table} SET {colum}='{value}' WHERE {colum}='{condition}' ".format(schema=schema
                                                                                                       , table=table,
                                                                                                       colum=colum,
                                                                                                       value=value,
                                                                                                       condition=condition)
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print(" update DB err", e)

    def deleteDB(self, schema, table, condition):
        sql = " delete from {schema}.{table} where {condition} ; ".format(schema=schema, table=table,
                                                                              condition=condition)
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except Exception as e:
            print("delete DB err", e)

if __name__ == "__main__":
    db = CRUD()
    current_time = datetime.now()
    db.insertDB(schema='myschema', table='sensor_data', colum='temp', data='temp')
    # db.save_sensor_data(current_time,temp, ph, do_value,con)
