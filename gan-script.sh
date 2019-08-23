cd src/

python main.py train --distributed --client &
sleep 5;
python main.py train --distributed --client &
sleep 5;
python main.py train --distributed --client &
sleep 5;
python main.py train --distributed --client &
sleep 10

python main.py train --distributed --master -f configuration/quickstart/mnist.yml

