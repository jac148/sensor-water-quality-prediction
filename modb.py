import minimalmodbus
import time
import dbconnect
import psycopg2
import sys
from datetime import datetime

# def check_error(value) :
#     v = int(value,16)
#     if v == 522 or v == 266 :
#         return True
#     else :
#         return False


def convert_register_value(value1, value2) :
    value2 = int(value2,16)
    value1 = int(value1,16)
    digit = value2 // 256
    value = value1 / (10 ** digit)

    return value

sensor_port = "/dev/tty.usbserial-10"
sensor_address_ph = 0x01
sensor_address_DO = 0x03
sensor_address_con = 0x04

now = time

try :
    # ph sensor connect
    instrument = minimalmodbus.Instrument(port=sensor_port, slaveaddress=sensor_address_ph)
    instrument.serial.baudrate = 9600
    instrument.serial.parity = minimalmodbus.serial.PARITY_NONE
    instrument.serial.bytesize = 8
    instrument.serial.stopbits = 1
    instrument.serial.timeout = 1

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



except Exception as e :
    print("Could not connect : ", e)

if __name__ == '__main__' :
    # try :
    #     response1 = instrument.read_register(0x0000, functioncode=int('0x04', 16))
    #     print("response data : ", hex(response1))
    try:
        flag = 0

        while True:
            # read ph sensor
            address_value_ph = []
            for register_address in range(16):
                num = register_address
                response_ph = instrument.read_register(register_address, functioncode=int('0x04', 16))
                # print(f"Register {register_address}: {hex(response_ph)}")
                address_value_ph.append(hex(response_ph))
                time.sleep(0.5)

                # if register_address == 2 :
                #     if not check_error(response_ph) :
                #         flag = 1
                #
                # if flag == 1 :
                #     address_value_ph[0], address_value_ph[1] = 0,0
                #     address_value_ph[8], address_value_ph[9] = 0,0
                #     break

            # print(address_value_ph)
            ph = convert_register_value(address_value_ph[0], address_value_ph[1])
            temp = convert_register_value(address_value_ph[8], address_value_ph[9])

            # print("ph : ", ph, "temp : ",temp)

            time.sleep(2)

            # read DO sensor
            address_value_DO = []
            for register_address in range(16):
                num = register_address
                response_DO = instrument_DO.read_register(register_address, functioncode=int('0x04', 16))
                # print(f"Register {register_address}: {hex(response_DO)}")
                address_value_DO.append(hex(response_DO))
                time.sleep(0.5)

                # if register_address == 2 :
                #     if not check_error(response_ph) :
                #         flag = 1
                #
                # if flag == 1 :
                #     address_value_DO[0], address_value_DO[1] = 0,0
                #     break


            # print(address_value_ph)
            do_value = convert_register_value(address_value_DO[0], address_value_DO[1])

            # print("DO : ", do)
            # time.sleep(2)

            print(now.strftime('%Y-%m-%d %H:%M:%S'))
            current_time = datetime.now()

            # read con sensor
            address_value_con = []
            for register_address in range(16):
                num = register_address
                response_con = instrument_con.read_register(register_address, functioncode=int('0x04', 16))
                # print(f"Register {register_address}: {hex(response_ph)}")
                address_value_con.append(hex(response_con))
                time.sleep(0.5)

                # if register_address == 2 :
                #     if not check_error(response_con) :
                #         flag = 1
                #
                # if flag == 1 :
                #     address_value_con[0], address_value_con[1] = 0,0
                #     break


            # print(address_value_ph)
            con = convert_register_value(address_value_con[0], address_value_con[1])
            print("ph : ", ph, "temp : ",temp)
            print("DO : ", do_value)
            print("con : ", con)


            # time.sleep(2)

            # db = dbconnect.CRUD()
            # db.save_sensor_data(current_time,temp, ph, do_value,con)

            # db = dbconnect.CRUD()
            # db.save_sensor_data(current_time, temp, ph, do_value,con)
            #
            # db.insertDB(schema='public', table='sensor_data', colum='ph', data=ph)
            # db.insertDB(schema='public', table='sensor_data', colum='temp', data=temp)
            # db.insertDB(schema='public', table='sensor_data', colum='do_value', data=do)


    except minimalmodbus.NoResponseError:
        print("No request")

    finally:
        instrument.serial.close()


    time.sleep(1)


