#!/bin/bash

cd ../

if [ ! -d "./logs" ]; then
	  mkdir ./logs
fi

# 使用不同的环境配置 TORNADO_PROFILES_ACTIVE dev | test | pro
if [ $TORNADO_PROFILES_ACTIVE ];
then
    echo "TORNADO_PROFILES_ACTIVE: $TORNADO_PROFILES_ACTIVE"
    cp conf/settings_$TORNADO_PROFILES_ACTIVE.py conf/settings.py
else
    echo "TORNADO_PROFILES_ACTIVE not found"
fi

# 编译silk-v3-decoder.tar.gz
tar -zxvf ./app/aidvoice/silk-v3-decoder.tar.gz -C ./app/aidvoice/
cd ./app/aidvoice/silk-v3-decoder/silk/
make && make decoder
cd -

# 创建临时存储语音文件的文件夹
if [ ! -d "./web/service/files" ]; then
      mkdir ./web/service/files
fi

case $1 in 
    start)
        if [ `ps -ef | grep driver.py | grep -v grep | wc -l` -ne 0 ]; then
            echo "process already exists ......"
        else
            echo "python driver.py"
            python driver.py -port=10080 2>&1 &
            echo "process start successfully ......"
        fi
        ;;
    stop)
        ps -ef | grep driver.py | grep -v grep | awk '{print $2}' | xargs kill -9
        echo "process stop successfully ......"
        ;;
    restart)
        ps -ef | grep driver.py | grep -v grep | awk '{print $2}' | xargs kill -9
        echo "process stop successfully ......"
        nohup python driver.py -port=10080 -log_file_prefix=./logs/tornado.log > ./logs/tornado.out 2>&1 &
        echo "process start successfully ......"
        ;;
    status)
        echo "active drivers: `ps -ef | grep driver.py | grep -v grep | wc -l`"
        ;;
    *)
        echo "$0 {start|stop|restart|status}"
        exit 4
        ;;
esac

tail -f /dev/null

