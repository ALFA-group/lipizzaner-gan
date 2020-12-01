cd src/

for i in {1..4..1}
do 
    echo "start client ${i}"
    python main.py train --distributed --client & sleep 2; 
done 

ps
wget http://0.0.0.0:5000/status
cat status 

# call the client killing script
netstat_output_path=../netstatOutputPath.txt 
sleep_time=8
port_path=../ports.txt 
# bash -x ../client_killing.sh ${netstat_output_path} ${sleep_time} ${port_path} &

python main.py train --distributed --master -f configuration/quickstart/toy-1d-gaussian.yml #mnist.yml