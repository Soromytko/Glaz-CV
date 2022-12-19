# Поиск похожих изоражений

## Зависимости
* pip install streamlit
* pip install opencv-python
* pip install pandas
* pip install sklearn
* pip install -U scikit-learn _(for KMeans)_
* pip install pathlib
* pip install tqdm

## Установка

### Windows
Запустить файл `install.bat` для установки необходимых пакетов

### Linux
Выполнить следующие команды:
* `chmod +x install.sh`
* `./install.sh`

## Запуск

Создать две директории: `training_data_set` и`image_database` В директорию `training_data_set` поместить изображения для обучения классификатора дескрипторов
([COCO128](www.kaggle.com/datasets/ultralytics/coco128)). В директорию `image_database` поместить изображения,
которые будут использоваться в качестве базы данных ([Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)). 

### Windows
По очереди запустить следующие файлы, дожидаясь окончания выполнения каждого из них:
* `train.bat` - тренировка модели.
* `init_database.bat` - индексирование базы данных.
* `run.bat` - запуск приложения с графическим интерфейсом.

### Linux
При первом запуске выполнить следующие команды, чтобы разрешить выполнение скриптов владельцу:
* `chmod +x train.sh`
* `chmod +x init_database.sh`
* `chmod +x run.sh`

При последующих запусках:
* `./train.sh` - тренировка модели.
* `./init_database.sh` - индексирование базы данных.
* `./run.sh` - запуск приложения с графическим интерфейсом.



