import minimalmodbus
import time
import dbconnect
import psycopg2
import sys
from datetime import datetime

def convert_register_value(value1, value2):
    value2 = int(value2, 16)
    value1 = int(value1, 16)
    digit = value2 // 256
    value = value1 / (10 ** digit)

    return value

def read_register1(instrument, arr) :
    for register_address in range(16):
        response = instrument.read_register(register_address, functioncode=int('0x04', 16))
        arr.append(hex(response))
        time.sleep(0.5)


sensor_port = "/dev/tty.usbserial-10"
sensor_address_ph = 0x01
sensor_address_DO = 0x03
sensor_address_con = 0x04

now = time

try:
    # ph sensor connect
    instrument_ph = minimalmodbus.Instrument(port=sensor_port, slaveaddress=sensor_address_ph)
    instrument_ph.serial.baudrate = 9600
    instrument_ph.serial.parity = minimalmodbus.serial.PARITY_NONE
    instrument_ph.serial.bytesize = 8
    instrument_ph.serial.stopbits = 1
    instrument_ph.serial.timeout = 1

    # DO sensor connect
    instrument_DO = minimalmodbus.Instrument(port=sensor_port, slaveaddress=sensor_address_DO)
    instrument_DO.serial.baudrate = 9600
    instrument_DO.serial.parity = minimalmodbus.serial.PARITY_NONE
    instrument_DO.serial.bytesize = 8
    instrument_DO.serial.stopbits = 1
    instrument_DO.serial.timeout = 1

    instrument_con = minimalmodbus.Instrument(port=sensor_port, slaveaddress=sensor_address_con)
    instrument_con.serial.baudrate = 9600
    instrument_con.serial.parity = minimalmodbus.serial.PARITY_NONE
    instrument_con.serial.bytesize = 8
    instrument_con.serial.stopbits = 1
    instrument_con.serial.timeout = 1



except Exception as e:
    print("Could not connect : ", e)

if __name__ == '__main__':
    try:
        flag = 0

        while True:
            try :
                # ----------------------------------------------------------------------------------------------------------------
                # read ph sensor
                address_value_ph = []
                read_register1(instrument_ph, address_value_ph)

                ph = convert_register_value(address_value_ph[0], address_value_ph[1])
                temp = convert_register_value(address_value_ph[8], address_value_ph[9])
                time.sleep(2)
                #-----------------------------------------------------------------------------------------------------------------

                # ----------------------------------------------------------------------------------------------------------------
                # read DO sensor
                address_value_DO = []
                read_register1(instrument_DO, address_value_DO)

                do_value = convert_register_value(address_value_DO[0], address_value_DO[1])
                time.sleep(2)
                # ----------------------------------------------------------------------------------------------------------------

                # ----------------------------------------------------------------------------------------------------------------
                # read con sensor
                address_value_con = []
                read_register1(instrument_con, address_value_con)

                con = convert_register_value(address_value_con[0], address_value_con[1])
                time.sleep(2)
                # ----------------------------------------------------------------------------------------------------------------

                current_time = datetime.now()
                print(current_time)
                print("ph : ", ph, "temp : ", temp)
                print("DO : ", do_value)
                print("con : ", con)

                db = dbconnect.CRUD()
                db.save_sensor_data(current_time,temp, ph, do_value,con)

            except Exception as e:
                print("ERROR : ", e)


    except minimalmodbus.NoResponseError:
        print("No request")

    finally:
        instrument_ph.serial.close()
        instrument_DO.serial.close()
        instrument_con.serial.close()

    time.sleep(1)

