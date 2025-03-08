from ultralytics import YOLO

#Cargar modelo
model = YOLO("yolo11n.pt") #Crea un modelo nuevo

#Usar el modelo

results = model.train(data="datap.yaml", epochs=1) #Entrena el modelo con los datos en data.yaml