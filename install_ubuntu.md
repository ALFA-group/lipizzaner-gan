# Lipizzaner

## Setup
```
git clone https://github.com/ALFA-group/lipizzaner-gan.git
cd lipizzaner-gan/
python3 --version
sudo apt-get update
sudo apt-get install python3-venv
python3 -m venv ~/my367
source ~/my367/bin/activate
sudo apt-get install python3-dev
sudo apt-get install gcc
pip install -r ./src/helper_files/requirements.txt
```

## MNIST
```
cd src/
python main.py train --distributed --client & sleep 5;
python main.py train --distributed --client & sleep 5;
python main.py train --distributed --client & sleep 5;
python main.py train --distributed --client &
ps
wget http://0.0.0.0:5000/status
cat status
python main.py train --distributed --master -f configuration/quickstart/mnist.yml
```

## Theoretical GAN
```
cd ../theoretical_experiments
sudo apt-get install python3-tk
python gaussian_gan.py
```

## Network traffic
```
cd ../src/data/network_data/
sudo apt-get install argus-client
./collect_network_traffic.sh
python analyze_network_file.py --pcap_file=network_capture.pcap --sequence_length=30
cd ../../
python main.py train --distributed --master -f configuration/lipizzaner-gan/network_traffic.yml
```
