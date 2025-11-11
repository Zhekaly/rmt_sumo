import traci
import csv
import os
import sys

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
CONFIG_FILE = r"C:\sumodemo\net\cross.sumocfg"

def collect_data(output_csv="traffic_data.csv"):
    """Собирает данные о трафике из SUMO симуляции"""
    
    if not os.path.exists(SUMO_BINARY):
        print(f"SUMO binary не найден: {SUMO_BINARY}")
        sys.exit(1)
    
    if not os.path.exists(CONFIG_FILE):
        print(f"Конфигурационный файл не найден: {CONFIG_FILE}")
        sys.exit(1)
    
    print("Запуск SUMO...")
    
    try:
        traci.start([
            SUMO_BINARY, 
            "-c", CONFIG_FILE,
            "--start",
            "--quit-on-end"
        ])
        
        print("SUMO успешно запущен!")
        
    except Exception as e:
        print(f"Ошибка при запуске SUMO: {e}")
        print("\nВозможные причины:")
        print("1. Проверьте, что cross.net.xml существует в папке net/")
        print("2. Пересоздайте сеть командой: netconvert -c net/cross.netccfg")
        print("3. Убедитесь, что все XML файлы корректны")
        sys.exit(1)
    
    step = 0
    max_steps = 3600
    fields = ["step", "lane_id", "veh_count", "waiting_time", "CO2"]
    data = []
    
    print(f"Сбор данных ({max_steps} шагов)...")
    
    try:
        while step < max_steps:
            traci.simulationStep()
            
            for lane_id in traci.lane.getIDList():
                veh_count = traci.lane.getLastStepVehicleNumber(lane_id)
                waiting_time = traci.lane.getWaitingTime(lane_id)
                
                CO2 = 0.0
                for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                    CO2 += traci.vehicle.getCO2Emission(veh_id)
                
                data.append([step, lane_id, veh_count, waiting_time, CO2])
            
            if step % 50 == 0:
                print(f"  Шаг {step}/{max_steps}")
            
            step += 1
        
        print("Симуляция завершена!")
        
    except Exception as e:
        print(f"Ошибка во время симуляции: {e}")
    finally:
        traci.close()
        print("SUMO закрыт")
    
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(data)
        
        print(f"Данные сохранены в {output_csv}")
        print(f"   Всего записей: {len(data)}")
        
    except Exception as e:
        print(f"Ошибка при сохранении данных: {e}")

if __name__ == "__main__":
    collect_data()