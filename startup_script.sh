cd src/
python main.py train --distributed --client & sleep 5; 
python main.py train --distributed --client & sleep 5;
python main.py train --distributed --client & sleep 5;
python main.py train --distributed --client &
ps

wget http://0.0.0.0:5000/status
cat status 
python main.py train --distributed --master -f configuration/quickstart/mnist.yml