cd src/
echo $PORT_UPPER_LIMIT
echo $NUM_CLIENTS_PER_NODE
h=$( hostname )

#sleep $JSM_NAMESPACE_RANK
printf "    - address: $h<>      port: 5000-$PORT_UPPER_LIMIT<>" >> configuration/quickstart/general.yml #      port: 5000-$PORT_UPPER_LIMIT\n" >> configuration/quickstart/general.yml 


for((i=1; i<=$NUM_CLIENTS_PER_NODE; i++))
do
  GPU_ID=$(($i % 6))
  export CUDA_VISIBLE_DEVICES=$GPU_ID; python main.py train --distributed --client &
  sleep 5;
done


if (( $JSM_NAMESPACE_RANK == 0 )); then

  sed -i 's/<>/\n/g' configuration/quickstart/general.yml

  printf "\n" >> configuration/quickstart/general.yml 
  python main.py train --distributed --master -f configuration/quickstart/mnist-gpu.yml
fi

wait