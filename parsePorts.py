# tcp        0      0 0.0.0.0:5000            0.0.0.0:*               LISTEN      14188/python        
# tcp        0      0 0.0.0.0:5001            0.0.0.0:*               LISTEN      14215/python        
# tcp        0      0 0.0.0.0:rfe             0.0.0.0:*               LISTEN      14229/python        
# tcp        0      0 0.0.0.0:5003            0.0.0.0:*               LISTEN      14242/python  
import sys

with open(sys.argv[1]) as f:
    lines = f.readlines()
f.close()

ports = []
for line in lines:
    splitUp = line.split()
    port = splitUp[3].split(":")
    port = port[1]
    if port == "rfe":
        ports.append("5002")
    else:
        ports.append(port)

outputFile = open(sys.argv[2], 'w+')
for port in ports:
    outputFile.write(port + "\n")
# outputFile.writelines(ports)
outputFile.close()
print(ports)

